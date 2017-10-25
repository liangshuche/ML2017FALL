import csv
import numpy as np
import sys

def readfile(path):
	f=open(path,'r')
	read=[row for row in csv.reader(f)]
	f.close()
	data=np.array(np.delete(read,0,0)).astype(np.int)
	return data

def normalization(data,mean,std):
	data_n=(data-mean)/std
	data_n[np.isnan(data_n)]=1
	return data_n

def sigmoid(x):
	x=np.clip(x,-100,100)
	z=1/(1+np.exp(-x))
	z=np.clip(z,0,0.999999999)
	return z

def gradient_descent(x_data,y_data,iteration):
	x_data=np.concatenate((np.ones((x_data.shape[0],1)),x_data),axis=1)
	w=np.zeros((len(x_data[0]),1))
	lr_w=np.zeros((len(x_data[0]),1))
	lr=1

	for i in range(iteration):

		hypo=np.dot(x_data,w)
		diff=sigmoid(hypo)-y_data

		w_grad=np.dot(x_data.T,diff)

		lr_w=lr_w+np.square(w_grad)

		w=w - lr/np.sqrt(lr_w) * w_grad

		if (i%200)==0:
			loss=-np.sum(y_data*np.log(sigmoid(hypo))+(1-y_data)*np.log(1-sigmoid(hypo)))
			print("iteration: %d | loss: %f \r" % (i,loss),end="")

	return w

def output(data,path,w):
	f=open(path,'w')
	f.write('id,label\n')
	data=np.concatenate((np.ones((data.shape[0],1)),data),axis=1)
	hypo=sigmoid(np.dot(data,w))
	#print (hypo)
	for i in range(len(data)):
		if hypo[i]>0.5:
			f.write("%d,1\n" % (i+1))
		else:
			f.write("%d,0\n" % (i+1))


#x_train=readfile("X_train")
x_train=readfile(sys.argv[1])
#y_train=readfile("Y_train")
#x_test=readfile("X_test")
x_test=readfile(sys.argv[2])
x_train_square=np.concatenate((x_train,x_train[:,0:6]**2),axis=1)
x_test_square=np.concatenate((x_test,x_test[:,0:6]**2),axis=1)
mean=np.mean(x_train_square,axis=0)
std=np.std(x_train_square,axis=0)
x_train_n=normalization(x_train_square,mean,std)
x_test_n=normalization(x_test_square,mean,std)

#w=gradient_descent(x_train_n,y_train,200000)
w=np.loadtxt('w')
#np.savetxt('w_20w_6feature',w)
output(x_test_n,sys.argv[3],w)
