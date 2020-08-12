from keras import models
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
import json
from keras.layers import Dense
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model
img_w = 480
img_h = 360
n_labels = 2

kernel = 3

encoding_layers = [
    Convolution2D(64, kernel, border_mode='same', input_shape=(img_h, img_w,16)),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(64, kernel, border_mode='same',),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),

    Convolution2D(128, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(128, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),

    Convolution2D(256, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(256, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(256, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),

    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),

    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),
]

autoencoder = models.Sequential()
autoencoder.encoding_layers = encoding_layers

for l in autoencoder.encoding_layers:
    autoencoder.add(l)
    #print(l.input_shape,l.output_shape,l)

decoding_layers = [
    UpSampling2D((2,2)),
    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D((2,2)),
    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(512, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(256, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),

    ZeroPadding2D(((1, 0), (0, 0))),

    UpSampling2D((2,2)),
    Convolution2D(256, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(256, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(128, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    
    UpSampling2D((2,2)),
    Convolution2D(128, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(64, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D((2,2)),
    Convolution2D(64, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(n_labels, 1, 1, border_mode='valid'),
    BatchNormalization(),
]
autoencoder.decoding_layers = decoding_layers
for l in autoencoder.decoding_layers:
    autoencoder.add(l)
    
print(autoencoder.summary())

#autoencoder.add(Reshape((n_labels, img_h*img_w)))
#autoencoder.add(Reshape((1, img_h*img_w, n_labels)))
#autoencoder.add(Permute((2, 1)))
autoencoder.add(Activation('softmax'))
print("ok1")
plot_model(autoencoder, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
with open('model_5l.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(autoencoder.to_json()), indent=2))
print("ok2")