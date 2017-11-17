import csv
import numpy as np
import sys

from random import shuffle

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, ZeroPadding2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

def transform_onehot(label,size):
	label_onehot=np.zeros((len(label),size))
	
	for i in range(len(label)):
		idx=int(label[i])
		label_onehot[i][idx]=1
	return label_onehot 

def readfile(path,norm,val,onehot_size,is_train):
	print ("Reading File...")
	f=open(path,'r')
	read=[row for row in csv.reader(f)]
	f.close()
	read.remove(read[0])
	if(1>val>0):
		shuffle(read)

	data_list=[]
	label_list=[]
	for row in read:
		data_row=row[1].split()
		data_list.append(data_row)
		label_list.append(row[0])
	
	if (norm):
		data=np.array(data_list).astype(np.float)/255.0
	else:
		data=np.array(data_list).astype(np.float)
		
	label=np.array(label_list).astype(np.float)
	
	if(is_train):
		label=transform_onehot(label,onehot_size)

	if(1>val>0):
		cut_idx=int(len(label)*val)
		data_train=data[cut_idx:]
		data_val=data[:cut_idx]
		label_train=label[cut_idx:]
		label_val=label[:cut_idx]
		return data_train,data_val,label_train,label_val
	elif(is_train):
		return data,label_onehot
	else:
		return data


x_train,x_val,y_train,y_val=readfile(sys.argv[1],255,0.1,7,True)
x_train=x_train.reshape(-1,48,48,1)
x_val=x_val.reshape(-1,48,48,1)

datagen=ImageDataGenerator(rotation_range=10,horizontal_flip=True,fill_mode='nearest')

dropout_rate=0.5

model = Sequential()

model.add(Conv2D(128,(5,5),padding='valid',activation='relu',input_shape=(48,48,1)))
model.add(Dropout(0.3))
model.add(ZeroPadding2D(padding=(1,1),data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(4,4),strides=(1,1)))
model.add(ZeroPadding2D(padding=(1,1),data_format='channels_last'))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(ZeroPadding2D(padding=(1,1), data_format='channels_last'))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Dropout(0.3))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2,2)))
model.add(ZeroPadding2D(padding=(1,1),data_format='channels_last'))

model.add(Conv2D(256,(3,3),activation='relu'))
model.add(ZeroPadding2D(padding=(1,1), data_format='channels_last'))

model.add(Conv2D(256,(3,3),activation='relu'))
model.add(Dropout(0.3))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2,2)))
model.add(ZeroPadding2D(padding=(1,1),data_format='channels_last'))

model.add(Conv2D(256,(3,3),activation='relu'))
model.add(ZeroPadding2D(padding=(1,1),data_format='channels_last'))

model.add(Conv2D(256,(3,3),activation='relu'))
model.add(ZeroPadding2D(padding=(1,1), data_format='channels_last'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2,2)))
model.add(Dropout(0.3))

model.add(Flatten())

for i in range(2):
	model.add(Dense(units=1024,activation='relu'))
	#model.add(BatchNormalization())	
	model.add(Dropout(dropout_rate))

model.add(Dense(units=7,activation='softmax'))

early_stop=EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
#'adam'
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
model.summary()
#model.fit(x_train,y_train,batch_size=64,epochs=200,validation_split=0.1,shuffle=True,callbacks=[early_stop])
model.fit_generator(datagen.flow(x_train,y_train,batch_size=64),steps_per_epoch=len(x_train)/64, epochs=200, verbose=1, callbacks=[early_stop], validation_data=(x_val,y_val), validation_steps=128)
model.save(sys.argv[2])
#y_test=model.predict( x_test, batch_size=128, verbose=0)
#np.save(output_path,y_test)

#result=model.evaluate(x_train,y_train)
#print ('acc:%f\n' % (result[1]))


