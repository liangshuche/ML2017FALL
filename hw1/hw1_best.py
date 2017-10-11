import csv
import numpy as np
import sys

def readfile_train(path):
	f=open(path,'r',encoding='big5')
	read=[row for row in csv.reader(f)]
	f.close()

	data_temp=np.array(np.delete(np.delete(read,0,0),[0,1,2],1))
	for i in range(10,4320,18):
		for j in range(24):
			if data_temp[i][j]=='NR':
				data_temp[i][j]=0

	data=np.zeros((18,5760))
	for i in range(4320):
		for j in range(24):
			x_idx=int(i%18)
			y_idx=int(np.floor_divide(i,18)*24+j)
			data[x_idx][y_idx]=float(data_temp[i][j])

	return data

def readfile_test(path):
	f=open(path,'r', encoding='big5')
	read=[row for row in csv.reader(f)]
	f.close()

	data_temp=np.array(np.delete(read,[0,1],1))
	for i in range(10,4320,18):
		for j in range(9):
			if data_temp[i][j]=='NR':
				data_temp[i][j]=0
	
	data=np.zeros((18,2160))
	for i in range(4320):
		for j in range(9):
			x_idx=int(i%18)
			y_idx=int(np.floor_divide(i,18)*9+j)
			data[x_idx][y_idx]=float(data_temp[i][j])
	
	return data

def create_train(data,n,row_s,row_e):
	month=12
	X=np.ones(((20*24-n)*month,n*(row_e-row_s)))
	Y=np.ones(((20*24-n)*month))
	for i in range(month):
		for j in range(20*24-n):
			X[i*(20*24-n)+j,:]=np.reshape(data[row_s:row_e,i*(20*24)+j:i*(20*24)+j+n],(1,n*(row_e-row_s)))
			Y[i*(20*24-n)+j]=data[9,i*(20*24)+j+n]

	#X=np.concatenate((X,X**2),axis=1)

	return X,Y

def create_test(data,n,row_s,row_e):
	data_test=np.zeros((240,n*(row_e-row_s)))
	for i in range(240):
		data_test[i,:]=np.reshape(data[row_s:row_e,(i+1)*9-n:i*9+9],(1,n*(row_e-row_s)))
	#data_test=np.concatenate((data_test,data_test**2),axis=1)
	return data_test

def normalization(data,mean,std):
	data_n=(data-mean)/std
	data_n[np.isnan(data_n)]=1
	return data_n

def gradient_descent(x_data,y_data,iteration,l):
	x_data=np.concatenate((np.ones((x_data.shape[0],1)),x_data),axis=1)
	w=np.zeros(len(x_data[0]))
	lr_w=np.zeros(len(x_data[0]))
	lr=1

	for i in range(iteration):

		hypo=np.dot(x_data,w)
		loss=hypo-y_data

		w_grad=2.0*np.dot(x_data.T,loss)+2*l*w
		w_grad[0]=w_grad[0]-2*l*w[0]

		lr_w=lr_w+np.square(w_grad)

		w=w - lr/np.sqrt(lr_w) * w_grad

		if (i%200)==0:
			print("iteration: %d | loss: %f \r" % (i,np.sum(np.square(loss))),end="")

	return w

def save_parameter(data,path):
	f=open(path,'w')
	for i in range(len(data)):
		f.write('%f\n' % data[i])

def read_parameter(path):
	f=open(path)
	data=[]
	for row in csv.reader(f):
		data.append(float(row[0]))

	print (data)

	data=np.array(data)
	return data

def output(data,path,w,n,row_s,row_e):
	f=open(path,'w')
	f.write('id,value\n')
	for i in range(240):
		#data_slice=np.ones(n*(row_e-row_s))
		#data_slice[:]=np.reshape(data[row_s:row_e,(i+1)*9-n:(i+1)*9],(1,n*(row_e-row_s)))
		f.write("id_%d,%f\n" % (i,np.dot(data[i,:],w[1:])+w[0]))

def cheat(x_data,y_data):
	return np.matmul(np.matmul(np.linalg.inv(np.matmul(x_data.T,x_data)),x_data.T),y_data)

def create_high_order_data(data,order):
	data_o=data
	for i in range(2,order+1):
		data_o=np.concatenate((data_o,data**i),axis=1)
	return data_o

#data_train=readfile_train("train.csv")
data_test=readfile_test(sys.argv[1])


#x_train,y_train=create_train(data_train,9,7,10)
x_test=create_test(data_test,9,7,10)

#mean=np.mean(x_train,axis=0)
#std=np.std(x_train,axis=0)
#save_parameter(mean,'mean_best.csv')
#save_parameter(std,'std_best.csv')
#print (mean)
#print (std)
#x_train=normalization(x_train,mean,std)
#x_test=normalization(x_test,mean,std)
#print (x_train_n)

#x_train_n=create_high_order_data(x_train,2)
x_test_n=create_high_order_data(x_test,2)
#print (x_train_n)
#print (x_test_n)
#np.savetxt('w_best_np',w)
w=np.loadtxt('w_best_np')
#mean=np.loadtxt('mean_best_np')
#std=np.loadtxt('std_best_np')
#w=gradient_descent(x_train_n,y_train,200000,0.0)
#save_w(w,'w_best.csv')
output(x_test_n,sys.argv[2],w,9,8,10)