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
val = 50

#testData = np.load('test.npy')
testData = pickle.load(open(sys.argv[1]+'test.p','rb'))
testData = np.asarray(testData['data'])
testData = np.reshape(testData, (10000,3,32,32)).astype('float32')/255.

json_file = open(sys.argv[2]+'.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(sys.argv[2]+".hdf5")

prediction2 = model.predict_classes(testData, batch_size = batchSize)
file_dir = sys.argv[3]
with open(file_dir, "w") as f:
    f.write("ID,class\n")
    for j in range(len(prediction2)):
        f.write(str(j)+","+str(prediction2[j])+"\n")
