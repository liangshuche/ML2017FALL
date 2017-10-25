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

def output(data,u1,u2,cov,path):
	f=open(path,'w')
	f.write('id,label\n')
	w=np.dot((u1-u2).T,np.linalg.inv(cov))
	b=-0.5*np.dot(np.dot(u1.T,np.linalg.inv(cov)),u1)+0.5*np.dot(np.dot(u2.T,np.linalg.inv(cov)),u2)+np.log(len(u1)/len(u2))
	w=np.append(b,w)
	#np.savetxt('w_gen',w)
	data=np.concatenate((np.ones((data.shape[0],1)),data),axis=1)
	hypo=sigmoid(np.dot(data,w))
	#print (hypo)
	for i in range(len(data)):
		if hypo[i]>0.5:
			f.write("%d,1\n" % (i+1))
		else:
			f.write("%d,0\n" % (i+1))


x_train=readfile(sys.argv[1])
y_train=readfile(sys.argv[2])
x_test=readfile(sys.argv[3])

mean=np.mean(x_train,axis=0)
std=np.std(x_train,axis=0)
x_train_class1=x_train_class2=normalization(x_train,mean,std)
x_test_n=normalization(x_test,mean,std)

class1_idx=[]
class2_idx=[]

for i in range(len(y_train)):
	if y_train[i]==0:
		class1_idx.append(i)
	else:
		class2_idx.append(i)

x_train_class1=np.delete(x_train_class1,class1_idx,0)
x_train_class2=np.delete(x_train_class2,class2_idx,0)


u1=np.mean(x_train_class1,axis=0)
u2=np.mean(x_train_class2,axis=0)

cov1=(np.dot((x_train_class1-u1).T,(x_train_class1-u1)))
cov2=(np.dot((x_train_class2-u2).T,(x_train_class2-u2)))
cov=(cov1+cov2)/(len(u1)+len(u2))

output(x_test_n,u1,u2,cov,sys.argv[4])
