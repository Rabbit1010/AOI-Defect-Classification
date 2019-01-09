# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 18:23:45 2018

@author: Wei-Hsiang, Shen
"""

from myReadFile import Read_CSV, Write_CSV

from keras.models import load_model
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import sys

def Predict(model, down_sampling_factor=8, output_csv_file_name='output2.csv'):
    
    # Read .csv file
    testing_data = Read_CSV('test.csv')
    img_file_path = './test_images'
        
    # Predict the image one-by-one
    for i in tqdm(range(1,testing_data.shape[0])):
        file_full_path = os.path.join(img_file_path, testing_data[i][0])
        im = Image.open(file_full_path)
        im = im.resize([int(512/down_sampling_factor), int(512/down_sampling_factor)])
        im = np.asarray(im)
        
        x = np.zeros([1,int(512/down_sampling_factor),int(512/down_sampling_factor),1])
        x[0,:,:,0] = im/255
        
        label = model.predict(x, batch_size=None, verbose=0, steps=None)
        
        testing_data[i][1] = np.argmax(label)
        
#        print("\rProgress {:2.1%}".format(i / testing_data.shape[0]), end="\r")

    Write_CSV(output_csv_file_name, testing_data)

if __name__ == '__main__':
    CNN_model = load_model('./checkpoints/model_all.h5')
    Predict(down_sampling_factor=4, model=CNN_model, output_csv_file_name='./checkpoints/1.8all.csv')    
    print("Finished!")