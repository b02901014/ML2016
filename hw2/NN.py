import numpy as np
import matplotlib.pyplot as plt
import csv
import pickle
import sys

validation = 1
epoch = 1000
size = 4001-validation
feSize = 57
nuNum = 40

xArray = np.ones((size,feSize+1))
yHat = np.zeros((size,1))
validX = np.ones((validation,feSize+1))
validY = np.zeros((validation,1))
weight1 = np.random.rand(feSize+1, nuNum)
weight2 = np.random.rand(nuNum, 1)
#weight1 = np.zeros((feSize+1, nuNum))
#weight2 = np.zeros((nuNum, 1))

#read training data
f = open(sys.argv[1],'r')
reader = csv.reader(f)
myList = list(reader)

myData = np.asarray(myList)
myData = np.delete(myData,0,1)
xArray[:,0:feSize] = np.delete(myData[0:size,:],feSize,1)
yHat[:,0] = myData[0:size,feSize]
validX = myData[size:size+validation,:]
validY = myData[size:size+validation,feSize]


xArray = xArray.astype('float32')
yHat = yHat.astype('float32')
validX = validX.astype('float32')
validY = validY.astype('float32')

a1 = np.zeros((size, nuNum))
a2 = np.zeros((size, 1))
grd1 = np.zeros((nuNum,1))
grd2 = np.zeros((feSize+1,nuNum))
lr = 0.01
r1 = 0.9
r2 = 0.999
epi = 10**-7
lamda = 0.01
moment1 = np.zeros((feSize+1, nuNum))
moment2 = np.zeros((nuNum,1))
v1 = 0
v2 = 0

for i in range(epoch):
    #forward
    a1 = 1/(1+np.exp(-np.dot(xArray, weight1)))
    a1[:, nuNum-1] = np.ones((size,)) #the last nu is the bias
    a2 = 1/(1+np.exp(-np.dot(a1, weight2)))

    #backward
    delta2 = a2*(1-a2)*-(yHat/a2+(yHat-1)/(1-a2))
    delta1 = a1*(1-a1)*np.dot(delta2, np.transpose(weight2))
    grd2 = np.dot(np.transpose(a1), delta2) + 2*lamda*weight2
    grd1 = np.dot(np.transpose(xArray), delta1) + 2*lamda*weight1

    #logistic
    moment1 = r1*moment1+(1-r1)*grd1 
    moment2 = r1*moment2+(1-r1)*grd2 
    v1 = r2*v1+(1-r2)*grd1*grd1
    v2 = r2*v2+(1-r2)*grd2*grd2
    mHat1 = moment1/(1-r1**(i+1))
    mHat2 = moment2/(1-r1**(i+1))
    vHat1 = v1/(1-r2**(i+1))
    vHat2 = v2/(1-r2**(i+1))
    weight1 = weight1 - lr/(np.sqrt(vHat1)+epi)*mHat1
    weight2 = weight2 - lr/(np.sqrt(vHat2)+epi)*mHat2
    #weight = weight -lr*grd    
    if i % 1 == 0:
        for j in range(len(a2)):
            if a2[j] <= 0.0000000000000001:
                a2[j] = a2[j]+0.0000000000000001
            if a2[j] >= 0.9999999999999999:
                a2[j] = a2[j]-0.0000000000000001
        loss = -((np.dot(np.transpose(yHat), np.log(a2))
                +np.dot(np.transpose(1-yHat), np.log(1-a2))))/size

        print 'loss '+str(i)+' '+str(loss)
        '''
        #validation
        correct = 0
        val = np.zeros((validation,1))
        aa1 = 1/(1+np.exp(-np.dot(validX, weight1)))
        aa1[:, nuNum-1] = np.ones((validX.shape[0],)) #the last nu is the bias
        aa2 = 1/(1+np.exp(-np.dot(aa1, weight2)))
        for j in range(aa2.shape[0]):
            if aa2[j] >= 0.5:
                val[j] = 1
            else:
                val[j] = 0
            if val[j] == validY[j]:
                correct = correct+1
                
        print 'val = '+str(((float)(correct)/validation))
        '''        
model = (weight1, weight2, nuNum, feSize)
pickle.dump(model, open(sys.argv[2],'wb+'))



