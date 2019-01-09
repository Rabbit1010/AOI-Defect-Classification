# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 14:37:05 2018

@author: Wei-Hsiang, Shen
"""

from myReadFile import Read_Training_Data

import numpy as np
import matplotlib.pyplot as plt
from plot_confusion_matrix import plot_confusion_matrix 
from resampling import ReSampling

import keras
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix

from CNN_Structure import CNN_Structure, CNN_Structure_Customized

def Train_Evaluate(X, y, down_sampling_factor=8, batch_size=50, epochs=20):
    num_classes = 6
    img_size_rows = int(512/down_sampling_factor)
    img_size_cols = int(512/down_sampling_factor)
    input_shape = (img_size_rows, img_size_cols, 1) # input_shape = (img_rows, img_cols, channel)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Image augmentation
    X_train, y_train = ReSampling(X_train, y_train, drawFig=True)
    X_test, y_test = ReSampling(X_test, y_test, drawFig=True)
    
    # Convert class vectors to binary class matrices
    y_train_onehot = keras.utils.to_categorical(y_train, num_classes)
    y_test_onehot = keras.utils.to_categorical(y_test, num_classes)
    
    model = CNN_Structure_Customized(input_shape, num_classes, conv2D_dropout=0.3, fc_dropout=0.5, conv2D_layer_count=2, regularize_C=0.01, fc_nodes=512)
    
    # define checkpoints
    filepath="./checkpoints/weights-improvement-{epoch:03d}-{val_acc:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    
    # Train the model
    history = model.fit(X_train, y_train_onehot, batch_size=batch_size,
              epochs=epochs, verbose=1, callbacks=callbacks_list,
              validation_data=(X_test, y_test_onehot))
    
    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # confusion matrix
    y_pred = model.predict(X_test, batch_size=50, verbose=0, steps=None)
    y_pred = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, [0,1,2,3,4,5], normalize=False)
    
    return model, history.history, cm

if __name__ == '__main__':
        
    # Read data according to .csv file
    print("Reading training data...")
    X, y = Read_Training_Data('./train.csv', './train_images', down_sampling_factor=4)

#    X_resampled, y_resampled = ReSampling(X, y, drawFig=False)
    
    model, metrics, cm = Train_Evaluate(X, y, down_sampling_factor=4, batch_size=64, epochs=25)

    model.save('./checkpoints/model.h5')
    
#    CNN_model = load_model('CNN_random_oversampling_4.h5')
#    Predict(down_sampling_factor=4, model=CNN_model, output_csv_file_name='output_random_oversampling.csv')    
    
#    cm = Kfold_Validation(X=X, y=y, down_sampling_factor=8, batch_size=50, epochs=20, fold_count=5)
#    Predict(down_sampling_factor=8, model_file_name='DIP_final_model.h5', output_csv_file_name='output2.csv')