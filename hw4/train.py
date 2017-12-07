from sys import argv
import numpy as np
import pickle as pk
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Embedding, Conv1D , LSTM, MaxPooling1D, BatchNormalization, GRU
from keras.layers import ZeroPadding1D, AveragePooling1D, Bidirectional
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

train_model=1
val_split=0.1
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipvalue=0.5)
epoch_num=20
model_path='model.h5'
train_label_path=argv[1]
w2v_model_path='w2v_256_model.bin'
embedding_dim=256
sequence_len=25


def readlabeldata(path):
    print('Reading Labeled Data...')
    data=[]
    label=[]
    with open(path) as f:
        for line in f:
            temp=line.strip().split('+++$+++')
            data.append(temp[1])
            label.append([int(temp[0])])
    return data,label

train_labeled_text,train_labeled_label=readlabeldata(train_label_path)

w2v_model=Word2Vec.load(w2v_model_path)
tokenizer=pk.load(open('tokenizer.pk','rb'))

vocab_size=len(tokenizer.word_index)+1
print (vocab_size)
x_train_labeled=tokenizer.texts_to_sequences(train_labeled_text)
x_train_labeled=pad_sequences(x_train_labeled,maxlen=sequence_len,padding='post',truncating='post')

x_train_labeled=np.array(x_train_labeled)
y_train_labeled=np.array(train_labeled_label)


if(1>val_split>0):
    val_idx=round(x_train_labeled.shape[0]*val_split)
    x=x_train_labeled[val_idx:][:]
    y=y_train_labeled[val_idx:][:]
    x_val=x_train_labeled[:val_idx][:]
    y_val=y_train_labeled[:val_idx][:]

print('Creating Embedding Layer...')
embedding_matrix=np.zeros((vocab_size, embedding_dim))
for word,i in tokenizer.word_index.items():
    if word in w2v_model:
        embedding_vector=w2v_model[word]
        if(embedding_vector) is not None:
            embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix[i] = embedding_matrix[0]

if (train_model):
    model = Sequential()
    model.add(Embedding(vocab_size,embedding_dim,weights=[embedding_matrix],input_length=sequence_len, trainable=False))
    model.add(GRU(256,activation='relu',recurrent_dropout=0.3, dropout=0.3,return_sequences=True))
    model.add(BatchNormalization())
    model.add(GRU(256,activation='relu',recurrent_dropout=0.3, dropout=0.3,return_sequences=True))
    model.add(BatchNormalization())
    model.add(GRU(256,activation='relu',recurrent_dropout=0.3, dropout=0.3))
    model.add(BatchNormalization())
    model.add(Dense(units=512,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(units=256,activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(units=1,activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['acc'])
	
    callbacks=[]
    checkpoint=ModelCheckpoint(model_path, save_best_only=True, verbose=0, monitor='val_acc', mode='max')
    early_stop=EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    callbacks.append(early_stop)
    callbacks.append(checkpoint)

    model.fit(x,y,batch_size=512,epochs=int(epoch_num),validation_data=(x_val,y_val),verbose=1,callbacks=callbacks)

