import numpy as np
import os
from sys import argv
from skimage import io
def read_img(path):
    dirs = os.listdir(path)
    data=np.zeros((415,1080000))
    for i,file_name in enumerate(dirs):
        data[i]=io.imread(path+'/'+file_name).flatten()
    return data

def mean(data):
    data = np.mean(data, axis=0).astype(np.uint8)
    output = data.reshape(600, 600, 3)
    io.imsave('p1.jpg',output)

def pca(data):
    data -= np.mean(data, axis=0)
    U, s, V = np.linalg.svd(data.T, full_matrices=False)
    
    U_re = U - np.min(U,axis=0)
    U_re /= np.max(U_re,axis=0)
    U_re = (U_re*255).astype(np.uint8)
    
    return U,U_re[:,0:4]

def reconstruct(data,eigen):
    mean=np.mean(data)
    img = data - mean
    
    w = np.dot(img,eigen)
    img_re = np.dot(eigen,w)
    
    img_re += mean
    img_re -= np.min(img_re)
    img_re /= np.max(img_re)
    img_re = (img_re*255).astype(np.uint8)
    
    io.imsave('reconstruction.jpg',img_re.reshape(600,600,3))
    return img_re


re_img=io.imread(os.path.join(argv[1],argv[2])).flatten()
data=read_img(argv[1])
eigen,face=pca(data)
reconstruct(re_img,eigen[:,0:4])

