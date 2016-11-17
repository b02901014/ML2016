import os
os.environ["TEANO_FLAGS"] = 'mode=FAST_RUN, device=gpu0, folatX=float32'
'''
import tensorflow as tf
tf.python.control_flow_ops = tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)
'''
from keras import backend as K
import numpy as np
import keras
import random
from keras.models import Sequential, model_from_json
from keras.layers import Input, Dense, Dropout, Convolution2D, MaxPooling2D, UpSampling2D, Activation, Flatten
from keras.models import Model
from keras.optimizers import Adam, Adadelta
from keras.preprocessing.image import ImageDataGenerator
import sys
import pickle


batchSize = 100
nbEpoch = 40
encodingDim = 32
inputImg = Input(shape=(3,32,32))
laSize = 5000
classNum = 10

def build_model():
    model = Sequential()
    model.add(Dense(64,input_shape=(128,)))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(classNum))
    model.add(Activation('softmax'))
    
    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


labelData = pickle.load(open(sys.argv[1]+'all_label.p','r'))
unlabelData = pickle.load(open(sys.argv[1]+'all_unlabel.p','r'))
#testData = pickle.load(open(sys.argv[1]+'test.p','r'))

labelData = np.reshape(np.asarray(labelData), (5000,3,32,32)).astype('float32')/255.
unlabelData = np.reshape(np.asarray(unlabelData), (45000,3,32,32)).astype('float32')/255.
#testData = np.reshape(np.asarray(testData), (10000,3,32,32)).astype('float32')/255.
label = np.zeros((laSize,classNum))
for i in range(laSize):
    label[i,i/(laSize/classNum)] = i/(laSize/classNum)
#########################################################
# split data into train and test parts
xTrain, yTrain = np.array([]), np.array([])
xVal, yVal = np.array([]), np.array([])
shuffle = np.random.permutation(laSize)
x = np.split(labelData[shuffle],classNum)
y = np.split(label[shuffle],classNum)
for i in range(classNum):
    da = np.split(x[i], [450, 500])
    la = np.split(y[i], [450, 500])
    if i==0 :
        xTrain, xVal = da[0], da[1]
        yTrain, yVal = la[0], la[1]
    else:
        xTrain = np.concatenate((xTrain,da[0]),axis=0)
        xVal = np.concatenate((xVal,da[1]),axis=0)
        yTrain = np.concatenate((yTrain,la[0]),axis=0)
        yVal = np.concatenate((yVal,la[1]),axis=0)
xTrain2 = np.concatenate((xTrain,unlabelData),axis=0)

x = Convolution2D(32,3,3,activation='relu',border_mode='same',dim_ordering='th')(inputImg)
x = MaxPooling2D((2,2))(x)
x = Convolution2D(16,3,3,activation='relu',border_mode='same')(x)
x = MaxPooling2D((2,2))(x)
x = Convolution2D(8,3,3,activation='relu',border_mode='same')(x)
encoded = MaxPooling2D((2,2))(x)

x = Convolution2D(8,3,3,activation='relu',border_mode='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Convolution2D(16,3,3,activation='relu',border_mode='same')(x)
x = UpSampling2D((2,2))(x)
x = Convolution2D(32,3,3,activation='relu',border_mode='same')(x)
x = UpSampling2D((2,2))(x)
decoded = Convolution2D(3,3,3,activation='sigmoid', border_mode='same')(x)

autoencoder = Model(inputImg, decoded)
adam = Adam(lr=0.001)
autoencoder.compile(optimizer=adam,loss='mse')
autoencoder.fit(xTrain2, xTrain2, nb_epoch=nbEpoch, batch_size=100, validation_data=(xVal,xVal))

encod = Model(inputImg, encoded)

model_json = encod.to_json()
with open('encod.json','w') as json_file:
    json_file.write(model_json)
encod.save_weights('encod.h5')

#feature of the autoencoder
print (encod.predict(xTrain).shape)
prediction = np.reshape(encod.predict(xTrain),(4500,128))
unprediction = np.reshape(encod.predict(unlabelData),(45000,128))
xVal = np.reshape(encod.predict(xVal),(500,128))
print (prediction.shape)
model = build_model()

for i in range (5):
    model.fit(prediction,yTrain,batch_size=batchSize, nb_epoch=nbEpoch, validation_data=(xVal, yVal), verbose=1)
    prob = model.predict_proba(unprediction)
    idx = np.argwhere(prob>0.99)

    shuffle2 = np.random.permutation(idx.shape[0])
    xPre = np.zeros((idx.shape[0],128))
    yPre = np.zeros((idx.shape[0],10))
    for i in range(idx.shape[0]):
        xPre[i] = unprediction[idx[i,0]]
        yPre[i,idx[i,1]] = 1
    yTrain2 = np.concatenate((yTrain,yPre),0)
    xTrain2 = np.concatenate((prediction,xPre),0)
    model.fit(xTrain2, yTrain2, batch_size=batchSize,nb_epoch=nbEpoch, validation_data=(xVal, yVal),verbose=1)


model_json = model.to_json()
with open(sys.argv[2]+'.json','w') as json_file:
    json_file.write(model_json)
model.save_weights(sys.argv[2]+'.h5')

