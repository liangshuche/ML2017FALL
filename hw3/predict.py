import csv
import numpy as np
import sys

from keras.models import load_model

model_path=['model_1.h5','model_2.h5','model_3.h5']
#model_path=['model_train_dnn.h5']
x_test=np.load('x_test.npy').reshape(-1,48,48,1)
prediction=0
for path in model_path:
	print ('Processing Model %s...' % (path))
	classifier=load_model(path)
	prediction+=classifier.predict(x_test)

#np.save('prediction.npy',prediction)
prediction=np.argmax(prediction,axis=1)
f=open(sys.argv[1],'w')
f.write('id,label\n')
for i in range(len(prediction)):
	f.write('%d,%d\n' % (i,prediction[i]))



'''

11101554
11150008
11132310


'''
