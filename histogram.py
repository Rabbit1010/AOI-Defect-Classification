# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 22:59:32 2018

@author: user
"""

from myReadFile import Read_Training_Data, Read_CSV, Write_CSV

import matplotlib.pyplot as plt
import pandas as pd

y = Read_CSV(csv_file_name='output5.csv')

pd.value_counts(y[:,1]).plot.bar()
plt.title('Class histogram before over-sampling')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()