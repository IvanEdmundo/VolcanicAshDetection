# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:21:08 2019

@author: ivan_
"""

import keras
#import cv2
import numpy as np
#import json
#np.random.seed(7) # 0bserver07 for reproducibility
#import os
#from sklearn.utils.class_weight import compute_class_weight

train_data = np.load('E://CNN//data//train_data.npy')
print (train_data.shape)
train_label = np.load('E://CNN//data//train_label.npy')
print (train_label.shape)
labeltr= np.argmax(train_label,axis=1)
print (labeltr.shape)

num_classes = 2


from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(train_data, labeltr)
from collections import Counter
print(sorted(Counter(y_resampled).items()))
train_label1 = keras.utils.to_categorical(y_resampled, num_classes)
print(X_resampled.shape)
print(y_resampled.shape)
print(train_label1.shape)
np.save('E://CNN//data//train_dataunder.npy',X_resampled)
np.save('E://CNN//data//train_labelunder.npy',y_resampled)
train_datau = np.load('E://CNN//data//train_dataunder.npy')
print (train_datau.shape)
train_labelu = np.load('E://CNN//data//train_labelunder.npy')
print (train_labelu.shape)