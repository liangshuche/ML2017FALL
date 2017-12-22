from sys import argv
import numpy as np
from keras.models import  load_model
from keras import backend as K

def readtestingdata(path):
    print('Reading Testing Data...') 
    userID,movieID=[],[]
    with open(path) as f:
        for line in f:
            temp=line.strip().split(',')
            userID.append(temp[1])
            movieID.append(temp[2])
    del userID[0],movieID[0]
    return np.array(userID),np.array(movieID)

def rmse(y_true,y_pred):
    return K.sqrt(K.mean(K.square(y_pred-y_true),axis=-1))

x_test_user,x_test_movie=readtestingdata(argv[1])
model=load_model('model/model_prob1.h5', custom_objects={'rmse':rmse})
result=model.predict([x_test_user,x_test_movie],batch_size=8192,verbose=1)
f=open(argv[2],'w')
f.write('TestDataID,Rating\n')
for i in range (result.shape[0]):
    f.write('%d,%f\n' % (i+1,result[i]))
f.close()

