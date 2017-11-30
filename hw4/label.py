from sys import argv
import numpy as np

from keras.models import load_model

#argv[1]:model_path
#argv[2]:confidence
#argv[3]:output_path

model=load_model(argv[1])
x_train=np.load('data/x_train_nolabel.npy')
result=model.predict(x_train,batch_size=256,verbose=1)
print (result)
'''
labeled_data=0
for i in range(x_train.shape[0]):
'''
