# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 13:53:59 2020

@author: ivan_
"""

import tensorflow as tf
import keras
import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import merge
from keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
from keras import backend as K
import matplotlib.pyplot as plt
#import cv2
import numpy as np
import json
np.random.seed(7) # 0bserver07 for reproducibility
import os
from sklearn.utils.class_weight import compute_class_weight
# Init wandb
import wandb
from wandb.keras import WandbCallback
wandb.init(project="mlp")
#class_weights = compute_class_weight('balanced', np.unique(y), y)

config = wandb.config
data_shape = 360*480
nb_epoch=100
#class_weighting= [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]
class_weighting= [1,1]
print ( 'ok1')
n_labels = 2
train_data = np.load('E://CNN//data//train_dataover.npy')
print (train_data.shape)
train_label = np.load('E://CNN//data//train_labelover.npy')
print (train_label.shape)
train_label1 = keras.utils.to_categorical(train_label, n_labels)
print(train_label1.shape)

print ( 'ok2')
test_data = np.load('E://CNN//data//test_data.npy')
test_label = np.load('E://CNN//data//test_label.npy')
print(test_data.shape)
print(test_label.shape)
print ( 'ok3')
#load the model:
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(16,)))
model.add(BatchNormalization())
model.add(Dropout(config.dropout))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(config.dropout))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(n_labels, activation='softmax'))

# COMPile and fit
adam= keras.optimizers.Adam(lr=config.learning_rate)
adadelta= keras.optimizers.Adadelta(lr=config.learning_rate)
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
print ( 'ok5')
# checkpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
print ( 'ok6')
history = model.fit(train_data,train_label1,callbacks=[WandbCallback()], batch_size=config.batch_size, epochs=nb_epoch,
                    verbose=1, class_weight = class_weighting,validation_data=(test_data, test_label), shuffle=True) # validation_split=0.33
print(history.history.keys())
print ( 'ok7')
model1 = model.evaluate(test_data, test_label, batch_size=nb_epoch)
print(model)


y_pred1 = model.predict_classes(test_data, batch_size=10000)

cm= confusion_matrix(np.argmax(test_label,axis=1), y_pred1)
print(cm)
print('Classification Report')
target_names = ['Ash', 'NoAsh']
print(classification_report(np.argmax(test_label,axis=1), y_pred1, target_names=target_names))



model.save_weights('E://CNN//data//model_weight_{}.hdf5'.format(nb_epoch))
print ( 'ok8')
# Save model to wandb
model.save(os.path.join(wandb.run.dir, "model.h5"))
print('ok9')