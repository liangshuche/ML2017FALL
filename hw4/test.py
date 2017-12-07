from sys import argv
import numpy as np
import pickle as pk
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

model_path='model.h5'
test_path=argv[1]
result_path=argv[2]
w2v_model_path='w2v_256_model.bin'
sequence_len=25

def readtestdata(path):
    print('Reading Test Data...')
    data=[]
    with open(path) as f:
        for line in f:
            temp=(line.strip().split(',',1))
            data.append(temp[1])
    del data[0]
    return data		

test_text=readtestdata(test_path)
w2v_model=Word2Vec.load(w2v_model_path)
tokenizer=pk.load(open('tokenizer.pk','rb'))

x_test=tokenizer.texts_to_sequences(test_text)
x_test=pad_sequences(x_test,maxlen=sequence_len,padding='post',truncating='post')

x_test=np.array(x_test)

model=load_model(model_path)
result=model.predict(x_test,batch_size=1024,verbose=1)

f=open(result_path,'w')
f.write('id,label\n')
for i in range (result.shape[0]):
    if(result[i]<0.5):
        f.write('%d,0\n' % (i))
    else:
        f.write('%d,1\n' % (i))
f.close()

