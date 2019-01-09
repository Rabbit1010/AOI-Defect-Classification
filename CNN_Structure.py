# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 02:20:41 2018

@author: Wei-Hsiang, Shen
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation

def CNN_Structure(input_shape, num_classes):
    # Building CNN
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
#    model.add(Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(Dense(128, activation='relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

#    model.summary()
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    
    return model

def CNN_Structure_Best(input_shape, num_classes):
    # Building CNN
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))

#    model.summary()
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    
    return model

def CNN_Structure_Customized(input_shape, num_classes, conv2D_dropout=0.25, fc_dropout=0.25, conv2D_layer_count=1, regularize_C=0.01, fc_nodes=128):
    # Building CNN
    model = Sequential()
    
    # First convolutional layers (with input shape specified)
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conv2D_dropout))
    
    # The other convolutional layers
    for _ in range(conv2D_layer_count-1):
        model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(conv2D_dropout))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(fc_nodes, activation='relu', kernel_regularizer=keras.regularizers.l2(regularize_C)))
    model.add(Dropout(fc_dropout))
    model.add(Dense(num_classes, activation='softmax'))

#    model.summary()
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    
    return model