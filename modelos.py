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


def get_img_rec(img, M):
    depth_model = utils.depth_model()
    im2r = utils.projected_image(img,  M, depth_model)
    
    return im2r

def get_model_2(img_shape):
    inputs = keras.Input(shape=img_shape)
    output = custom_Res_2(inputs)
    
    if inputs.shape[0] == None:
        imr = inputs[:,:,:,0]
    else:
        imr = get_img_rec(inputs[:,:,:,:3], output)
    
    model = tf.keras.Model(inputs=inputs, outputs=[output, imr])
    
    return model

##################################################################################################################
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

def conv_block(x, filters, k, pooling = False, striding = True, norm = False):
    if norm:
      x = layers.BatchNormalization()(x) 
    if pooling:
        x = layers.MaxPooling2D(2, padding="same")(x)
        
    if striding:
        x = layers.Conv2D(filters, k, activation= "relu", padding="same", strides=(2,2))(x)
           
    return x
    
def up_block(x, filters, k, norm = False):   
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest")(x)
    if norm:
      x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, k, activation="relu", padding="same")(x)
    
    return x

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
    
    
    #outputs = 0.01 * layers.Dense(last_output, kernel_initializer=initializer)(x)
    output_1 = 0.01 * layers.Dense(last_output, kernel_initializer=initializer)(x)
    output_2 = 0.001 * layers.Dense(last_output, kernel_initializer=initializer)(x)
    
    outputs = tf.concat([output_1, output_2], axis = -1)
   
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


