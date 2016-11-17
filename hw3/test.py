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
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Convolution2D, MaxPooling2D, UpSampling2D, Activation, Flatten
from keras.models import Model
from keras.models import model_from_json
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

testData = pickle.load(open(sys.argv[1]+'test.p','rb'))
testData = np.asarray(testData['data'])
testData = np.reshape(np.asarray(testData), (10000,3,32,32)).astype('float32')/255.
json_file = open('encod.json','r')
loaded_model_json = json_file.read()
json_file.close()
encod = model_from_json(loaded_model_json)
encod.load_weights('encod.h5')
test = np.reshape(np.asarray(encod.predict(testData)),(10000,128))

json_file = open(sys.argv[2]+'.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(sys.argv[2]+'.h5')
prediction3 = model.predict_classes(test, batch_size=batchSize)
file_dir = sys.argv[3]
with open(file_dir, "w") as f:
    f.write("ID,class\n")
    for j in range(len(prediction3)):
        f.write(str(j)+","+str(prediction3[j])+"\n")

