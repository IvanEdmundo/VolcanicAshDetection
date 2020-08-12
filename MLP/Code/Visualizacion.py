# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 13:28:48 2019

@author: ivan_
"""
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
data_shape = 360*480

#class_weighting= [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]
class_weighting= [9,1]
print ( 'ok1')
# load the data

n_labels = 2

train_data = np.load('G://CNN//data//train_data.npy')

print (train_data.shape)
train_label = np.load('G://CNN//data//train_label.npy')

print (train_label.shape)


print ( 'ok2')
test_data = np.load('G://CNN//data//test_data.npy')
test_label = np.load('G://CNN//data//test_label.npy')

print(test_data.shape)
print(test_label.shape)
print('Training data shape : ', train_data.shape, train_label.shape)
 
print('Testing data shape : ', test_data.shape, test_label.shape)
 
# Find the unique numbers from the train labels
classes = np.unique(train_label)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)
 
plt.figure(figsize=[10,5])
 
# Display the first image in training data
plt.subplot(121)
plt.imshow(train_data[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(train_label[0]))
 
# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_data[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(test_label[0]))