import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
'''
import tensorflow as tf
tf.python.control_flow_ops = tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.1
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)
'''
import numpy as np
import pickle
import random
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam, SGD, Adadelta
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
import sys

# Constant define
batchSize = 150
classNum = 10
nbEpoch = 40
nbEpoch2 = 25
rows, cols, channels = 32, 32, 3
val = 50
laSize = 5000

def build_model():
    # Sequential
    model = Sequential()
    model.add(Convolution2D(64, 3, 3,border_mode='same', input_shape=(3,32,32),dim_ordering='th'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3,border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
     
    model.add(Convolution2D(256, 3, 3,border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    # classification
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classNum))
    model.add(Activation('softmax'))
    # compile
    adam = Adam(lr=0.0005)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model
 

###########################################################
# load all label
labelData = pickle.load(open(sys.argv[1]+'all_label.p','rb'))
unlabelData = pickle.load(open(sys.argv[1]+'all_unlabel.p','rb'))
labelData = np.reshape(labelData, (5000,3,32,32)).astype('float32')/255.
unlabelData = np.reshape(unlabelData, (45000,3,32,32)).astype('float32')/255.
#testData = np.reshape(np.asarray(np.load('test.npy')), (10000,3,32,32)).astype('float32')/255.
#unlabelData = np.concatenate((unlabelData,testData),0)
label = np.zeros((laSize,classNum))
for i in range(laSize):
    label[i,i/(laSize/classNum)] = i/(laSize/classNum)
#########################################################
# split data into train and test parts
xTrain, yTrain = np.array([]), np.array([])
xVal, yVal = np.array([]), np.array([])
# x_test, y_test = np.array([]), np.array([])
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
#################################################
# build model
model = build_model()

#################################################
# Image Augmentation
train_datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
train_datagen.fit(xTrain)
val_datagen = ImageDataGenerator()
train_datagen.fit(xTrain)

Val = train_datagen.flow(xVal, yVal, batch_size=batchSize)
size = 0

#model.load_weights("selftrain.hdf5")

for i in range(5):
    #model.fit(xTrain,yTrain,batch_size=batchSize,nb_epoch=nbEpoch,validation_data=(xVal,yVal),shuffle=True)
    #model.fit_generator(Train, samples_per_epoch=45000, nb_epoch=nbEpoch,callbacks=[early_stop],validation_data=(xVal,yVal))
    
    Train = train_datagen.flow(xTrain, yTrain, batch_size=batchSize)
    filepath=sys.argv[2]+".hdf5"
    early_stopping = EarlyStopping(monitor='val_loss',patience=15)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc',verbose=1,save_best_only=True,mode='max')
    callbacks_list = [checkpoint, early_stopping]    
    model.fit_generator(Train, samples_per_epoch=75000, nb_epoch=nbEpoch, callbacks=callbacks_list, validation_data=(xVal, yVal))

    model_json = model.to_json()
    with open(sys.argv[2]+".json", "w") as json_file:
        json_file.write(model_json)

##################################################

    yPre = np.array([])
    xPre = np.array([])
    prob = model.predict_proba(unlabelData)
    idx = np.argwhere(prob>0.999)
    size = idx.shape[0]
    xPre = np.zeros((size,3,32,32))
    yPre = np.zeros((size,10))

    for i in range(size):
        xPre[i] = unlabelData[idx[i,0]]
        yPre[i, idx[i,1]] = 1  
    xTrain2 = np.concatenate((xTrain, xPre), axis=0)
    yTrain2 = np.concatenate((yTrain, yPre), axis=0)
    Train = train_datagen.flow(xTrain2, yTrain2, batch_size=batchSize)

    early_stopping = EarlyStopping(monitor='val_loss',patience=10)
    #model.fit_generator(Train, samples_per_epoch=75000, nb_epoch=nbEpoch, callbacks=callbacks_list, validation_data=(xVal, yVal))
    model.fit_generator(Train, samples_per_epoch=60000, nb_epoch=nbEpoch2, callbacks=callbacks_list, validation_data=(xVal, yVal))

#####################################################################
