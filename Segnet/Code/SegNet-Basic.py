
import wandb
from wandb.keras import WandbCallback
wandb.init(project="segnet-volcanic-ash")

import tensorflow as tf
import keras.models as models
from keras.layers.core import Layer, Dense, Dmportropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional i Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
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
from sklearn.metrics import classification_report, confusion_matrix
#class_weights = compute_class_weight('balanced', np.unique(y), y)

data_shape = 360*480

#class_weighting= [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]
class_weighting= [1,1]
print ( 'ok1')
# load the data



train_data = np.load('E://MODIS//data//train_data.npy')
print (train_data.shape)
train_label = np.load('E://MODIS//data//train_label.npy')
print (train_label.shape)
#labeltr= np.argmax(train_label,axis=1)
#print (labeltr.shape)
#from imblearn.under_sampling import RandomUnderSampler
print ( 'ok2')
test_data = np.load('E://MODIS//data//test_data.npy')
test_label = np.load('E://MODIS//data//test_label.npy')
print(test_data.shape)
print(test_label.shape)
print ( 'ok3')
# load the model:
with open('E://MODIS//prog_labelInteger8//model_5l.json') as model_file:
    segnet_basic = models.model_from_json(model_file.read())
print ( 'ok4')

segnet_basic.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])
print ( 'ok5')
# checkpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

nb_epoch = 100
#epochs = 100
batch_size = 3
print ( 'ok6')
'''

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# Fit the model'''


#from keras.utils import plot_model
#plot_model(segnet_basic, to_file='model.png')

#from imblearn.keras import balanced_batch_generator
#training_generator, steps_per_epoch = balanced_batch_generator(train_data, train_label, sampler=RandomUnderSampler(), batch_size=3, random_state=42)

history = segnet_basic.fit(train_data, train_label, batch_size= batch_size ,callbacks=[WandbCallback()], epochs=nb_epoch,
                    verbose=1, class_weight = class_weighting , validation_data=(test_data, test_label), shuffle=True) # validation_split=0.33
print(history.history.keys())
avg = np.mean(history.history['acc'])
print('The Average Training Accuracy is', avg)
avg1 = np.mean(history.history['val_acc'])
print('The Average Validation Accuracy is', avg1)


#fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
print ( 'ok7')
model = segnet_basic.evaluate(test_data, test_label, batch_size=nb_epoch)
print(model)


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#y_pred1 = segnet_basic.predict_classes(test_data, batch_size=1000)

#cm= confusion_matrix(np.argmax(test_label,axis=1), y_pred1)
#print(cm)
#print('Classification Report')
#target_names = ['Ash', 'NoAsh']
#print(classification_report(np.argmax(test_label,axis=1), y_pred1, target_names=target_names))

# This save the trained model weights to this file with number of epochs
segnet_basic.save_weights('E://MODIS//data//model_weight_{}.hdf5'.format(nb_epoch))
print ( 'ok8')
segnet_basic.model.save(os.path.join(wandb.run.dir, "model.h5"))
print ( 'ok9')