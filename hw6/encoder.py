import numpy as np
import pandas as pd
from sys import argv
from keras.models import Model, load_model
from sklearn.cluster import KMeans
train=0

x=np.load(argv[1])
x=(x-np.mean(x,axis=0))/255

encoder=load_model('encoder_model')
embeddings = encoder.predict(x)
k_means = KMeans(n_clusters=2).fit(embeddings)
predict = k_means.labels_

f = pd.read_csv(argv[2])
IDs, idx1, idx2 = np.array(f['ID']), np.array(f['image1_index']), np.array(f['image2_index'])

o = open(argv[3],'w')
o.write('ID,Ans\n')
for idx, i1, i2 in zip(IDs, idx1, idx2):
    pred=int(predict[i1]==predict[i2])
    o.write('%d,%d\n' % (idx, pred))
o.close()
