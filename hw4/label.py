from sys import argv
import numpy as np

from keras.models import load_model

#argv[1]:model_path
#argv[2]:confidence
#argv[3]:output_path

model=load_model(argv[1])
x_train=np.load('data/x_train_nolabel.npy')
result=model.predict(x_train,batch_size=512,verbose=1)
#print (result)

x_train_semi=[]
y_train_semi=[]

for i in range(x_train.shape[0]):
	if(result[i][0]>float(argv[2])):
		x_train_semi.append(x_train[i])
		y_train_semi.append([1,0])
	if(result[i][1]>float(argv[2])):
		x_train_semi.append(x_train[i])
		y_train_semi.append([0,1])

x_train_semi=np.array(x_train_semi)
y_train_semi=np.array(y_train_semi)

np.save('data/x_train_semi.npy',x_train_semi)
np.save('data/y_train_semi.npy',y_train_semi)
