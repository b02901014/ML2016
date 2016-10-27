import numpy as np
import csv
import pickle
import sys


model = pickle.load(open('model', 'rb'))
weight = model[0]
feSize = model[1]

# running test data
f = open('spam_test.csv','r')
reader = csv.reader(f)
myList = list(reader)
myData = np.asarray(myList)

testX = np.ones((myData.shape[0],feSize))
myData = np.delete(myData,0,1)
testX[:,0:feSize-1] = myData
print testX.shape
print weight.shape

testX = testX.astype('float32')
print testX

file_dir = sys.argv[3]
with open(file_dir, 'wb') as f:
    f.write("id,label\n")
    for j in range(testX.shape[0]):
        testY = np.dot(testX[j,:],weight)
        testY = 1/(1+np.exp(-testY))

        if testY >= 0.5:
            value = 1
        else:
            value = 0
        f.write(str(j+1)+","+str(value)+"\n")

