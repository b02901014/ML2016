import numpy as np
import csv
import pickle
import sys

# running test data
model = pickle.load(open(sys.argv[1], 'rb'))
weight1 = model[0]
weight2 = model[1]
nuNum = model[2]
feSize = model[3]

f = open(sys.argv[2],'r')
reader = csv.reader(f)
myList = list(reader)
myData = np.asarray(myList)

testX = np.ones((myData.shape[0],feSize+1))
myData = np.delete(myData,0,1)
testX[:,0:feSize] = myData

testX = testX.astype('float32')

file_dir = sys.argv[3]
with open(file_dir, 'wb') as f:
    f.write("id,label\n")
    print np.dot(testX, weight1)
    aa1 = 1/(1+np.exp(-np.dot(testX, weight1)))
    aa1[:, nuNum-1] = np.ones((testX.shape[0],)) #the last nu is the bias
    aa2 = 1/(1+np.exp(-np.dot(aa1, weight2)))
    for j in range(aa2.shape[0]):
        if aa2[j] >= 0.5:
            value = 1
        else:
            value = 0
        f.write(str(j+1)+","+str(value)+"\n")

