import numpy as np
import matplotlib.pyplot as plt
import csv
import pickle
import sys

epoch = 500000
size = 4001
feSize = 58

xArray = np.ones((size,feSize))
yHat = np.zeros((size,1))
bias = np.ones((size,1))
#weight = np.zeros((feSize,1))
weight = 0.001*np.random.random_sample((58,1))

f = open(sys.argv[1],'r')
reader = csv.reader(f)
myList = list(reader)

myData = np.asarray(myList)
myData = np.delete(myData,0,1)
xArray[:,0:feSize-1] = np.delete(myData,feSize-1,1)
yHat[:,0] = myData[:,feSize-1]

xArray = xArray.astype('float32')
yHat = yHat.astype('float32')
#linear regression

lr = 0.00007
loss = 0
y = 0
lamda = 0.01

file_dir = "weightInit.csv"
with open(file_dir, 'wb') as f:
    for j in range(feSize):
        f.write(str(float(weight[j,:]))+"\n")


r1 = 0.9
r2 = 0.999
epi = 10**-7
moment = np.zeros((feSize,1))
v = 0
total = 0

for i in range(epoch):
    y = 1/(1+np.exp(-np.dot(xArray,weight)))
    grd = np.dot(np.transpose(xArray),(y - yHat))+2*lamda*weight
    '''
    moment = r1*moment+(1-r1)*grd 
    v = r2*v+(1-r2)*grd*grd
    mHat = moment/(1-r1**(i+1))
    vHat = v/(1-r2**(i+1))
    weight = weight - lr/(np.sqrt(vHat)+epi)*mHat
    '''
    total = total + np.sum(grd**2)/size
    ada = np.sqrt(total/(1+i))
    weight = weight -lr/ada*grd    
    if i % 1000 == 0:
        print str(i)+' '+str(loss)
        for j in range(len(y)):
            if y[j] <= 0.0000000000000001:
                y[j] = y[j]+0.0000000000000001
            if y[j] >= 0.9999999999999999:
                y[j] = y[j]-0.0000000000000001
        loss = -(np.dot(np.transpose(yHat), np.log(y))
                +np.dot(np.transpose(1-yHat), np.log(1-y))-
                lamda*np.dot(np.transpose(weight),weight))/size


model = (weight, feSize)
pickle.dump(model, open(sys.argv[2], 'wb+'))

