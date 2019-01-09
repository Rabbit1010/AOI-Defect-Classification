# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 15:01:38 2018

@author: Wei-Hsiang, Shen
"""

import csv
import numpy as np
import os
from tqdm import tqdm
from PIL import Image

def Read_Training_Data(csv_file_name='train.csv', img_file_path='./train_images', down_sampling_factor=8):
    with open(csv_file_name, newline='') as csvfile:
        rows = csv.reader(csvfile)
        
        csvArray = []
        # Read CSV file row by row
        for row in rows:
            # Store it into an array
            csvArray = np.append(csvArray, np.asarray(row))
        
    # Delete the first row
    csvArray = np.delete(csvArray, (0,1), axis=0)

    # Reshape the csvArray array to [csvArray_count,2]
    data_count = int(len(csvArray)/2)
    csvArray = np.reshape(csvArray, [data_count,2])
    
    data_filename = csvArray[:,0]
    
    # Data label
    label = csvArray[:,1]
    label = label.astype(int)
    
    # Read image to 3D array
    data = np.zeros([data_count,int(512/down_sampling_factor),int(512/down_sampling_factor),1])
    for i in tqdm(range(data_count)):
        file_full_path = os.path.join(img_file_path, data_filename[i])
        im = Image.open(file_full_path)
        im = im.resize([int(512/down_sampling_factor), int(512/down_sampling_factor)])
        data[i,:,:,0] = np.asarray(im)
    
    # Scale the data to 0~1
    data /= 255
    
    return data, label

def Read_CSV(csv_file_name='test.csv'):
    with open(csv_file_name, newline='') as csvfile:
        rows = csv.reader(csvfile)
        
        csvArray = []
        # Read CSV file row by row
        for row in rows:
            # Store it into an array
            csvArray = np.append(csvArray, np.asarray(row))
        
    data_count = int(len(csvArray)/2)
    csvArray = np.reshape(csvArray, [data_count,2])
    
    return csvArray

def Write_CSV(file_name, csv_array):
    # Write array into .csv file
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
    
        for i in range(csv_array.shape[0]):
            writer.writerow([csv_array[i][0],csv_array[i][1]])