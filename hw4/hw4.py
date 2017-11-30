from sys import argv
import numpy as np

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Embedding, Conv1D , LSTM, MaxPooling1D, BatchNormalization
from keras.layers import ZeroPadding1D, AveragePooling1D
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences

opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
epoch_num=argv[1]
model_path=argv[2]
output_path=argv[3]
train_label_path='data/training_label.txt'
train_nolabel_path='data/training_nolabel.txt'
test_path='data/testing_data.txt'


def readlabeldata(path):
	print('Reading Labeled Data...')
	data=[]
	label=[]
	with open(path) as f:
		for line in f:
			temp=line.strip().split(' +++$+++ ')
			data.append(temp[1])
			if (temp[0]=='0'):
				label.append([1,0])
			else:
				label.append([0,1])
	
	return data,label


#def readnonlabeldata:

def readtestdata(path):
	print('Reading Test Data...')
	data=[]
	with open(path) as f:
		for line in f:
			temp=line.strip().split(',')
			data.append(temp[1])
			
	del data[0]
	#print (data)
	return data		

train_labeled_text,train_labeled_label=readlabeldata(train_label_path)
test_text=readtestdata(test_path)

text_target=list(set().union(train_labeled_text,test_text))

tokenizer=Tokenizer()
tokenizer.fit_on_texts(text_target)

x_train_labeled=tokenizer.texts_to_sequences(train_labeled_text)
x_test=tokenizer.texts_to_sequences(test_text)

sequence_len=max(max(len(i) for i in x_train_labeled),max(len(j) for j in x_test))
vocab_size=len(tokenizer.word_index)+1
print("Total Unique Words: %d, Max Sequence Length: %d" % (vocab_size,sequence_len))

x_train_labeled=pad_sequences(x_train_labeled,maxlen=sequence_len, padding='post')
x_test=pad_sequences(x_test,maxlen=sequence_len, padding='post')


x_train_labeled=np.array(x_train_labeled)
#print (train_labeled_label)
y_train_labeled=np.array(train_labeled_label)
#print (y_train_labeled)
#print (x_train_labeled)
#print (tokenizer.texts_to_sequences(line1))
x_test=np.array(x_test)


model = Sequential()
model.add(Embedding(vocab_size,128))
model.add(Conv1D(64,5,activation='relu'))
model.add(Dropout(0.1))
model.add(ZeroPadding1D())
model.add(AveragePooling1D(3))
model.add(BatchNormalization())
model.add(Conv1D(128,5,activation='relu'))
model.add(Dropout(0.1))
model.add(ZeroPadding1D())
model.add(MaxPooling1D(2))
model.add(BatchNormalization())
model.add(Conv1D(128,5,activation='relu'))
model.add(Dropout(0.1))
model.add(ZeroPadding1D())
model.add(MaxPooling1D(2))
model.add(BatchNormalization())
model.add(LSTM(256,activation='sigmoid', dropout=0.2, return_sequences=True))
model.add(LSTM(256,activation='sigmoid', dropout=0.2, return_sequences=True))
model.add(LSTM(256,activation='sigmoid', dropout=0.2))
model.add(Dense(units=1024,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(units=1024,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(units=2,activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['acc'])
	
callbacks=[]
checkpoint=ModelCheckpoint(model_path, save_best_only=True)
early_stop=EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
callbacks.append(early_stop)
callbacks.append(checkpoint)

model.fit(x_train_labeled,y_train_labeled,batch_size=128,epochs=int(epoch_num),validation_split=0.1,verbose=1,callbacks=callbacks)

model=load_model(model_path)
result=model.predict(x_test,batch_size=128,verbose=1)
np.save(output_path,result)

