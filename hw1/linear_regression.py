import numpy as np
import matplotlib.pyplot as plt
import csv

iteration = 500000
size = 5652;
allData = np.zeros((18,5760))

xArray = np.ones((size,163))
yHat = np.zeros((size,1))
bias = np.ones((size,1))
weight = np.zeros((163,1))


f = open('train.csv','r')
reader = csv.reader(f)
myList = list(reader)

myData = np.asarray(myList)
myData = np.delete(myData,[0,1,2],1)
myData = np.delete(myData,0,0)

for i in range(240):
    for j in range(24):
        if myData[10+18*i,j] == 'NR':
            myData[10+18*i,j]=0

index = 0
for i in range(240):
    temp = myData[18*i:18+18*i, 0:24];
    temp = temp.astype('float32')
    allData[:,index:index+24] = temp
    index = index+24


index2 = 0
for i in range(12):
    for j in range(471):
        idx = j+i*471
        if idx==5652:
            break
        xArray[idx,0:162] = np.reshape(allData[:, index2:index2+9],(1,162))
        yHat[idx,:] = allData[9,index2+9]
        index2 = index2+1
    index2 = index2+9

#linear regression
lr = 0.0000000003
landa = 0
loss = 0
y = 0

t = np.arange(0., iteration/1000)
tt = np.array(t)

for i in range(iteration):
    y = np.dot(xArray,weight)
    grd = 2*np.dot(np.transpose(xArray),(y-yHat))+2*landa*weight
    loss = (np.dot(np.transpose(y-yHat),(y-yHat))+landa*np.dot(np.transpose(weight),weight))/5652
    weight = weight - grd*lr

# running test data
f = open('test_X.csv','r')
reader = csv.reader(f)
myList = list(reader)
myData = np.asarray(myList)

testX = np.ones((myData.shape[0]/18,163))
for i in range(myData.shape[0]/18):
    for j in range(9):
        if myData[10+18*i,j+2] == 'NR':
            myData[10+18*i,j+2] = 0
    testX[i,0:162] = np.reshape(myData[18*i:18*(i+1),2:11].astype('float32'),(1,162))

file_dir = "linear_regression.csv"
with open(file_dir, 'wb') as f:
    f.write("id,value\n")
    for j in range(testX.shape[0]):
        testY = np.dot(testX[j,:],weight)
        f.write("id_"+str(j)+","+str(float(testY))+"\n")

