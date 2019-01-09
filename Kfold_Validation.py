# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 15:34:50 2019

@author: Wei-Hsiang, Shen
"""

from myReadFile import Read_Training_Data, Write_CSV

import numpy as np
import random

import keras
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold

from CNN_Structure import CNN_Structure_Customized
from resampling import ReSampling

def Kfold_Validation(down_sampling_factor=8, batch_size=50, epochs=20, fold_count=5,
                     conv2D_dropout=0.25,
                     fc_dropout=0.25,
                     conv2D_layer_count=1,
                     regularize_C=0.01,
                     fc_nodes=128):
     # Read data according to .csv file
    print("Reading training data...")
    X, y = Read_Training_Data('./train.csv', './train_images', down_sampling_factor=down_sampling_factor)
    
    num_classes = 6
    img_size_rows = int(512/down_sampling_factor)
    img_size_cols = int(512/down_sampling_factor)
    input_shape = (img_size_rows, img_size_cols, 1) # input_shape = (img_rows, img_cols, channel)
    
    # Cross validation scores
    cv_scores = []
    # Max validation accuracy
    max_val_accs = []
    # Max validation accuracy index
    max_val_indexs = []
    # Training history
    historys = []

    # Split data into training and validation set
    kfold = StratifiedKFold(n_splits=fold_count, shuffle=False)    
    i_fold = 0
    for train, test in kfold.split(X, y):
        model = CNN_Structure_Customized(input_shape=input_shape, num_classes=num_classes,
                                         conv2D_dropout=conv2D_dropout,
                                         fc_dropout=fc_dropout,
                                         conv2D_layer_count=conv2D_layer_count,
                                         regularize_C=regularize_C,
                                         fc_nodes=fc_nodes)
        
        # Image augmentation
        X_train, y_train = ReSampling(X[train], y[train], drawFig=False)
        X_test, y_test = ReSampling(X[test], y[test], drawFig=False)
        
        # Convert class vectors to binary class matrices
        y_onehot_train = keras.utils.to_categorical(y_train, num_classes)
        y_onehot_test = keras.utils.to_categorical(y_test, num_classes)
        
        # define checkpoints
        filepath_front = "./checkpoints/down{}-batch{}-conv_dr{}-fc_dr{}-layer{}-C{}-fc_nd{}-fold{}-".format(down_sampling_factor,
                    batch_size,conv2D_dropout, fc_dropout, conv2D_layer_count, regularize_C, fc_nodes, i_fold)
        filepath=filepath_front + "epoch{epoch:02d}-val_acc{val_acc:.4f}.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        
        # Train the model
        history = model.fit(X_train, y_onehot_train, batch_size=batch_size,
                            epochs=epochs, verbose=0, callbacks=callbacks_list,
                            validation_data=(X_test, y_onehot_test))
        
    	# Evaluate the model
        scores = model.evaluate(X_test, y_onehot_test, verbose=0)
        print("%s: %.4f%%" % (model.metrics_names[1], scores[1]*100))
        
        cv_scores.append(scores[1] * 100)
        historys.append(history.history['val_acc'])
        max_val_accs.append(np.max(history.history['val_acc']))
        max_val_indexs.append(np.argmax(history.history['val_acc']))
        i_fold += 1
        
    print("K-fold average validation accuracy : %.4f%% (+/- %.4f%%)" % (np.mean(cv_scores), np.std(cv_scores)))
    
    return historys, cv_scores, max_val_accs, max_val_indexs

if __name__ == '__main__':   
    # Random search to optimize hyperparameters
    # total grid = 2*3*3*3*3*2*3 = 1296 grids
    grid_down_sampling_factor = random.choice([4]) # 4, 8
    grid_batch_size = random.choice([64]) # 16, 32, 64
    grid_epochs = 55
    grid_conv2D_dropout = random.choice([0.3]) # 0, 0.125, 0.25
    grid_fc_dropout = random.choice([0.5]) # 0, 0.25, 0.5
    grid_conv2D_layer_count = random.choice([2]) # 1, 2
    grid_regularize_C = random.choice([0.01]) # 0.01, 0
    grid_fc_nodes = random.choice([128]) # 64, 128, 256
    metrics, cv_scores, max_val_accs, max_val_indexs = Kfold_Validation(down_sampling_factor=grid_down_sampling_factor, 
                                                                        batch_size=grid_batch_size, 
                                                                        epochs=grid_epochs,
                                                                        conv2D_dropout=grid_conv2D_dropout,
                                                                        fc_dropout=grid_fc_dropout,
                                                                        conv2D_layer_count=grid_conv2D_layer_count,
                                                                        regularize_C=grid_regularize_C,
                                                                        fc_nodes=grid_fc_nodes)
    
#    Write_CSV('./checkpoints/random_search.csv', csv_array)