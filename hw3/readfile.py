import csv
import numpy as np
import sys

def readfile(path):
	print ("Reading File...")
	f=open(path,'r')
	read=[row for row in csv.reader(f)]
	f.close()
	read.remove(read[0])

	data_list=[]
	for row in read:
		data_row=row[1].split()
		data_list.append(data_row)
	
	data=np.array(data_list).astype(np.float)/255.0
	return data

x_test=readfile(sys.argv[1])
np.save('x_test.npy',x_test)
