# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:17:22 2019

@author: MOCTE
"""

from __future__ import absolute_import
from __future__ import print_function

#import cv2
#from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.io import loadmat
from keras.preprocessing import image
import tensorflow as tf

from scipy import stats




#from helper import *
import os

# Copy the data to this dir here in the SegNet project /CamVid from here:
# https://github.com/alexgkendall/SegNet-Tutorial
DataPath = 'E:\\MODIS\\'
DataPath2 = 'E:\\MODIS\\data\\'
#DataPath = '.\\SegNet\\CamVid\\'
data_shape = 360*480*16*4

def bytes_to_int(bytes):
    result = 0

    for b in bytes:
        result = result * 256 + int(b)

    return result

# GRADED FUNCTION: normalizeRows
def normalized(x):
    
    
    norm=np.zeros((16,360, 480),np.float32)
    

  
    for i in range(16):
        min_H= min_norm=1000
        max_H= max_norm=-1000
        H=x[i,:,:]
        max_H = np.max(H)
        min_H = np.min(H)
        norm[i,:,:] = stats.zscore(H)
        max_norm = np.max(norm[i,:,:])
        min_norm = np.min(norm[i,:,:])
        print ('Min_h, max_h, min_norm, max_norm',min_H, max_H, min_norm, max_norm)
        norm[i,:,:]= (norm[i,:,:]- min_norm) /(max_norm-min_norm)
        
   
    return norm
 



def one_hot_it(labels):
    x=[]
    lab1=[]
    lab2=[]
   
    
    lab1 = labels <= 0.0  #ceniza de volcan
    x.append(lab1)
    lab2 = labels > 0.0 # ceniza no humo
    x.append(lab2)
    
 
    #x = np.zeros([2,360,480],dtype=int)
    
    '''for i in range(360):
        #print(i)
        for j in range(480):
            val=0
            if labels[i][j]<=0.000: val=1
            x[val][i][j]=1'''
           
    return x


def one_hot_it2(labels):
    
    
    lin=360
    col=480
    
    x=[]
    #lab1=[]
    #lab2=[]
    #lab1=np.array((360,480),dtype=np.int32)
    #lab2=np.array((360,480),dtype=np.int32)
    
    lab1 = labels <= -0.2  #ceniza de volcan
    x.append(lab1)
    print('label', lab1[100,100])
    
    
    
    lab1 = labels > 2 # ceniza no humo
    print('label', lab1[100,100])
    x.append(lab1)
   
    #x2 = np.array(x,dtype=np.uint8)
    return np.array(x,dtype=np.uint8) 
    






#img= cv2.imread('0001TP_006690.jpg',1)
    
#img= cv2.imread('H:\\2-COURSOS\\remotesensingdata\\Keras-SegNet-Basic-master\\0001TP_006690.jpg',0)
#cv2.imshow('image',img)
#plt.imshow(img)
#cv2.waitKey(0)



def load_data(mode):
      
    data = []
    x3=[]
   
    label = []
    Directory= DataPath + mode
    with open(DataPath + mode +'.txt') as f:
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]
        print ('txt',txt)
        print ('len(txt)',len(txt))
        
    for i in range(len(txt)):
        #print(os.getcwd() +'\\SegNet\\CamVid\\'+ txt[i][0][15:])
        #print(os.getcwd() +'\\SegNet\\CamVid\\'+ mode + '\\'+ txt[i][0][21:])
        #print(os.getcwd() +'\\SegNet\\CamVid\\'+ mode + 'annot\\'+ txt[i][0][21:])
        
        #print( txt[i][1][7:][:-1])
        #img_path1 = (os.getcwd() +'\\SegNet\\CamVid\\'+ mode + '\\'+ txt[i][0][19:])
        #img_path2 = (os.getcwd() +'\\SegNet\\CamVid\\'+ mode + 'annot\\'+ txt[i][0][19:])
        img_path1 = (Directory+'\\'+ txt[i][0][0:16]+'\\'+ txt[i][0][0:16]+'EmisividadGeoCorte')
        img_path2 = (Directory+'\\'+ txt[i][0][0:16]+'\\'+ txt[i][0][0:16]+'b31_b32')
       
        print('i,img_path1',i,img_path1)
        print('i,img_path2',i,img_path2)
       
         
        f1 = open(img_path1, 'rb')
        f2 = open(img_path2, 'rb')
        
        x1=np.fromfile(f1,dtype=np.float32)
        x2=np.fromfile(f2,dtype=np.int32)
        
        #data=np.array(data,dtype=np.float32)
        dim_size=np.array((360,480,16),dtype=np.int32)
        
        print('data.shape',x1.shape)
        print('type data',type(x1))
        print('type image', type(dim_size))
        
        #imagen1 = x1.reshape(dim_size[0],dim_size[1],dim_size[2])
        #imagen2 = x2.reshape(dim_size[0],dim_size[1])
        
        imagen1 = x1.reshape(16,360,480)
        imagen2 = x2.reshape(360,480)
       
        imagen1 = normalized(imagen1)
        
        
        print('image.shape',imagen1.shape)
        print('type image', type(imagen1))
        
        print("Size of the array: ", imagen1.size)
        print("Length of one array element in bytes: ", imagen1.itemsize)
        print("Total bytes consumed by the elements of the array: ", imagen1.nbytes)

        
        #imagen = data.reshape((16,360,480))
        
                
        #img1 = image.load_img(img_path1,grayscale=False)
        #x1 = image.img_to_array(img1)
        #x2=np.load(img_path1, mmap_mode='r')
        #img1 = image.load_img(img_path1)
        #img2 = image.load_img(img_path2)
        
        #img2 = image.load_img(img_path2,grayscale=False)
        #x1 = image.img_to_array(img1, data_format='channels_first')#print(img1.shape)
        #x1 = image.img_to_array(img1)
        #print("ok1")
        #print(x1.shape)
        #x2 = image.img_to_array(img2)[:,:,0]
        print("ok2")
        
        print("ok4")
        print(imagen1.shape[0], 'canales')
        print(imagen1.shape[1], 'lineas')
        print(imagen1.shape[2], 'columnas')
        
        #print((imagen[0][0][0]))
        #print((imagen[0][0][1]))
        #print((imagen[0][0][2]))
        #print((imagen[0][0][3]))
       
        print("ok43")

        
        
        #print(imagen[0:1][0][0])
        img=imagen1[0,0:360,0:480]
        plt.matshow(img)
        plt.show()
        print("ok44")
        
        img=imagen2[0:360,0:480]
        plt.matshow(img)
        plt.show()
        print("ok44")
        #plt.imshow(img)
        #plt.show()
       # print("ok5")
        
        print('X1 SHAPE', x1.shape)
        #x2 = (img2)
        #print(type(x2))
        #x2.astype(int)
        #print(x2.dtype)
       
        print('X2 SHAPE',x2.shape)
        #plt.imshow(x1)
        #plt.imshow(x2)
        
 
        
        
        print('OK1')
        #data=np.append(data, imagen1)
        print('before np.rollaxis(imagen1,0,2) shape:',imagen1.shape)
        x3=np.rollaxis(imagen1,0,3)
        print('after np.rollaxis(imagen1,0,2) shape:',x3.shape)
        data.append(x3)
        #data.append(np.rollaxis(imagen1,2))
        #imagen3=x3.reshape(16,360,480)
        #data.append(imagen1)
        print("Total bytes of data: ", len(data))
        
        #data.append(np.rollaxis(normalized(x1),2))
        print('OK2')
        #print(type(label))
        #label= np.append(label,one_hot_it(imagen2))
        #label.append(one_hot_it(imagen2))
        print('X4 before np.rollaxis(,0,2) shape:',imagen2.shape)
        x4= one_hot_it2(imagen2)
        
        #print('X4 before np.rollaxis(,0,2) shape:',x4.shape)
          
        label.append(x4)
        print("Total bytes of data: ", len(label))
        
        
        #print('OK3')
        #data.append(np.rollaxis(normalized(cv2.imread(os.getcwd() + txt[i][0][7:])),2))
        
        #label.append(one_hot_it(cv2.imread(os.getcwd() + txt[i][1][7:][:-1])[:,:,0]))

        #label.append(one_hot_it(cv2.imread(os.getcwd() + txt[i][1][7:][:-1])[:,:,0]))
        print('.',end='')
        
  
    
        
       

       # data= np.reshape(dat, (360,480,16))
  
       
        
        
    
    return np.array(data), np.array(label)

modo= "train"
modo1= "test"
data, label = load_data(modo)
data1, label1 = load_data(modo1)


img_path_salida1 = (DataPath2+modo+'_data.npy')
img_path_salida2 = (DataPath2+modo+'_label.npy')
print(img_path_salida1)
print(img_path_salida2)
print('X4 before np.rollaxis(,0,2) shape:',data.shape)
print('X4 before np.rollaxis(,0,2) shape:',label.shape)
xlabel=np.rollaxis(label,1,4)
print('X4 before np.rollaxis(,0,2) shape:',xlabel.shape)

with open(img_path_salida1, "wb") as F1:
    np.save(F1,data)
    
with open(img_path_salida2, "wb") as F2:
    np.save(F2,xlabel) 
    
print( DataPath2+modo+'_data.npy')
train_data = np.load(DataPath2+modo+'_data.npy')
print ('fin',train_data.shape)

img6=train_data[0,0:360,0:480,15]
plt.matshow(img6)
plt.show()
print("ok444")

print( DataPath2+modo+'_data.npy')
label_data = np.load(DataPath2+'train_label.npy')
print ('fin',label_data.shape)

img6=label_data[0,0:360,0:480,1]
plt.matshow(img6)
plt.show()
print("ok444")

img6=label_data[0,0:360,0:480,0]
plt.matshow(img6)
plt.show()
print("ok555")

img_path_salida3 = (DataPath2+modo1+'_data.npy')
img_path_salida4 = (DataPath2+modo1+'_label.npy')
print(img_path_salida3)
print(img_path_salida4)
print('X4 before np.rollaxis(,0,2) shape:',data1.shape)
print('X4 before np.rollaxis(,0,2) shape:',label1.shape)
xlabel1=np.rollaxis(label1,1,4)
print('X4 before np.rollaxis(,0,2) shape:',xlabel1.shape)

with open(img_path_salida3, "wb") as F3:
    np.save(F3,data1)
    
with open(img_path_salida4, "wb") as F4:
    np.save(F4,xlabel1) 
    
print( DataPath2+modo1+'_data.npy')
test_data = np.load(DataPath2+modo1+'_data.npy')
print ('fin',test_data.shape)

img10=test_data[0,0:360,0:480,15]
print('dimension',img10.shape)
plt.matshow(img10)
plt.show()
print("ok666")

print( DataPath2+modo1+'_data.npy')
label1_data = np.load(DataPath2+'test_label.npy')
print ('fin',label1_data.shape)

img10=label1_data[0,0:360,0:480,1]
plt.matshow(img10)
plt.show()
print("ok666")

img11=label1_data[0,0:360,0:480,0]
plt.matshow(img11)
plt.show()
print("ok666")


