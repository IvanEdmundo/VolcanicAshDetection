# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:08:20 2019

@author: ivan_
"""
import keras
#import cv2
import numpy as np

#np.random.seed(7) # 0bserver07 for reproducibility
train_data = np.load('E://CNN//data//train_data.npy')
print (train_data.shape)
train_label = np.load('E://CNN//data//train_label.npy')
print (train_label.shape)
labeltr= np.argmax(train_label,axis=1)
print(labeltr.shape)
num_classes = 2
from imblearn.over_sampling import SMOTE
ros = SMOTE(random_state=42)
X_resampled, y_resampled = ros.fit_resample(train_data,labeltr)
from collections import Counter
print(sorted(Counter(y_resampled).items()))
train_label1 = keras.utils.to_categorical(y_resampled,num_classes)
print(y_resampled.shape)
print(train_label1.shape)

np.save('E://CNN//data//train_datasmote.npy',X_resampled)
np.save('E://CNN//data//train_labelsmote.npy',y_resampled)
train_datao = np.load('E://CNN//data//train_datasmote.npy')
print (train_datao.shape)
train_labelo = np.load('E://CNN//data//train_labelsmote.npy')
print (train_labelo.shape)