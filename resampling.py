# -*- coding: utf-8 -*-
"""
Since the training data is very unbalanced, we want to oversampling the small 
categories by image augmentation.

@author: Wei-Hsiang, Shen
"""

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from skimage.transform import AffineTransform, warp

import random
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

def OverSampling(X, y):
    """
    OverSampling to make the imblanced dataset have same size on each categories.
    """
    data_count = X.shape[0]
    img_row = X.shape[1]
    img_column = X.shape[2]
       
    ros = RandomOverSampler(random_state=0)
    X_2 = np.resize(X, [data_count, img_row*img_column])
    X_oversampled, y_oversampled = ros.fit_resample(X_2, y)
    
    X_oversampled = np.resize(X_oversampled, [X_oversampled.shape[0], img_row, img_column, 1])

    # Unit testing
    if X_oversampled.shape[0]!=y_oversampled.shape[0]:
        raise ValueError("Inconsistent shape between X and y")
    if X_oversampled.shape[3]!=1:
        raise ValueError("Channel should be 1 (gray-scale image)")

    return X_oversampled, y_oversampled

def Image_Shift(image, vector=[random.randint(-13,13),random.randint(-13,13)]):
    """
    Randomly shift image to increase data size.
    """
    transform = AffineTransform(translation=vector)
    shifted = warp(image, transform, mode='symmetric', preserve_range=True)

    shifted = shifted.astype(image.dtype)
    return shifted

def ReSampling(X, y, drawFig=False):
    """
    Image Augmentation by horizontally and vertically flipping all images to 4x the image count.
    """
#    X_oversampled, y_oversampled = OverSampling(X, y)
    
    print("Augmenting image data... (flipping)")
    X_fliped = np.tile(X, [4,1,1,1]) # 4x the image count
    y_fliped = np.tile(y, 4) # 4x the image count
    for i in tqdm(range(X_fliped.shape[0])):
        if i>=0 and i<X_fliped.shape[0]/4:
            continue
        elif i>=X_fliped.shape[0]/4 and i<X_fliped.shape[0]/2:
            X_fliped[i,:,:,0] = cv2.flip(X_fliped[i,:,:,0], 1) # Horizontally flip
        elif i>=X_fliped.shape[0]/2 and i<X_fliped.shape[0]/4*3:
            X_fliped[i,:,:,0] = cv2.flip(X_fliped[i,:,:,0], 0) # Vertically flip
        elif i>=X_fliped.shape[0]/4*3 and i<X_fliped.shape[0]:
            X_fliped[i,:,:,0] = cv2.flip(X_fliped[i,:,:,0], -1) # Horizontally + vertically flip
        
    print("Augmenting image data... (shifting)")
    X_shifted = np.tile(X_fliped, [4,1,1,1]) # 4x the image count
    y_shifted = np.tile(y_fliped, 4) # 4x the image count
    for i in tqdm(range(X_shifted.shape[0])):
        if i>=0 and i<X_shifted.shape[0]/4:
            continue
        else:
            X_shifted[i,:,:,0] = Image_Shift(X_shifted[i,:,:,0])
            
#    plt.imshow(X_shifted[49,:,:,0])
#    plt.show()
#    plt.imshow(X_shifted[99,:,:,0])
#    plt.show()
#    plt.imshow(X_shifted[149,:,:,0])
#    plt.show()
#    plt.imshow(X_shifted[199,:,:,0])
#    plt.show()
    
    # Draw histogram of classes
    if drawFig == True:
        plt.figure()
        pd.value_counts(y).plot.bar()
        plt.title('Class histogram before re-sampling')
        plt.xlabel('Class')
        plt.ylabel('Frequency')
        plt.show()
        
        plt.figure()
        pd.value_counts(y_shifted).plot.bar()
        plt.title('Class histogram after re-sampling')
        plt.xlabel('Class')
        plt.ylabel('Frequency')
        plt.show()
    
    # Unit testing
    if X_shifted.shape[0]!=y_shifted.shape[0]:
        raise ValueError("Inconsistent shape between X and y")    
    if X_shifted.shape[3]!=1:
        raise ValueError("Channel should be 1 (gray-scale image)")
        
    return X_shifted, y_shifted