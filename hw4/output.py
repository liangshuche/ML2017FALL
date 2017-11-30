import numpy as np
import sys

result=np.load(sys.argv[1])

#print (result)
result=np.argmax(result,axis=1)


f=open(sys.argv[2],'w')
f.write('id,label\n')
for i in range(len(result)):
   f.write('%d,%d\n' % (i,result[i]))