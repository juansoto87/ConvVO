import numpy as np
from tensorflow import keras
from keras import layers
import tensorflow as tf
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import os
import pretrained_models as pre
import utils

'''
This file contains the functions and models used during training
'''

# Creation of convolutional residual blocks, requires specifying the number of filters f, filter size k,
# pooling or sriding and whether to use batch normalization.

def residual_block(x, filters, k, pooling = False, striding = True, norm = False, std = False):
    residual = x
    x = layers.Conv2D(filters, k, activation= "relu", padding="same")(x)
    x = layers.Conv2D(filters, k, activation="relu", padding="same")(x)
    if pooling:
        x = layers.MaxPooling2D(2, padding="same")(x)
        residual = layers.Conv2D(filters, 1, strides=2)(residual)
    if striding:
        x = layers.Conv2D(filters, k, activation= "relu", padding="same", strides=2)(x)
        residual = layers.Conv2D(filters, 1, strides=2)(residual)
    elif filters != residual.shape[-1]:
        residual = layers.Conv2D(filters, 1)(residual)
    x = layers.add([x, residual])
    if norm:
      x = layers.BatchNormalization()(x)
    return x

#Creation of convolutional blocks by defining number of filters f, filter size k,
# pooling or striding and normalization.

def conv_block(x, filters, k, pooling = False, striding = True, norm = False):
    if norm:
      x = layers.BatchNormalization()(x) 
    if pooling:
        x = layers.MaxPooling2D(2, padding="same")(x)
        
    if striding:
        x = layers.Conv2D(filters, k, activation= "relu", padding="same", strides=(2,2))(x)
           
    return x
    
# Creation of dense layers, required to define number of neurons n, dropout d_o,
# activation function, initializer and normalization.
def fully_custom(x, n, d_o, activation, initializer, norm = False):
    x = layers.Dense(n, kernel_initializer=initializer)(x)
    if norm:
    	x = layers.BatchNormalization()(x)
    x = layers.Dropout(d_o)(x)
    if activation == 'lrelu':
    	x = layers.LeakyReLU()(x)
    elif activation == 'tanh':
    	x = tf.keras.activations.tanh(x)
    return x


# Final Model
def custom_Res_06(img_shape):
    #depth_model = utils.depth_model()
    inputs = keras.Input(shape=img_shape)
    x = inputs
    x = x / 255
    x = residual_block(x, 128, 7, pooling= False, striding = True, norm = True)
    x = residual_block(x, 256, 5, pooling= False, striding = True, norm = True)
    x = residual_block(x, 512, 3, pooling= False, striding = True, norm = True)
    x = residual_block(x, 512, 3, pooling= False, striding = True, norm = True)

    x = layers.Conv2D(512, 1, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 1, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 1, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 1, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 1, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    initializer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1, seed=None)
    activation = 'relu'
    
    x = layers.Flatten()(x)
    lx = x.shape[-1]
    d_o = 0.3
    
    x = fully_custom(x, 2048, d_o, activation, initializer, norm = True)
    x = fully_custom(x, 1024, d_o, activation, initializer, norm = True)
    x = fully_custom(x, 512, d_o, activation, initializer, norm = True)
    x = fully_custom(x, 128, d_o, activation, initializer, norm = True)
    x = fully_custom(x, 64, d_o, activation, initializer, norm = True)
    x = fully_custom(x, 32, 0, activation, initializer, norm = True)
   

    last_output = 3
    

    output_1 = 0.01 * layers.Dense(last_output, kernel_initializer=initializer)(x)
    output_2 = 0.001 * layers.Dense(last_output, kernel_initializer=initializer)(x)
    
    outputs = tf.concat([output_1, output_2], axis = -1)
   
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


