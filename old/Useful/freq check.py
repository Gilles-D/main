# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:21:08 2023

@author: MOCAP
"""
# import cv2
import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt


rootdir = r'D:\SOD_2023\0014_24_05\1'


test=[]

for subdirs, dirs, files in os.walk(rootdir):
    if re.search(r'\d+$', subdirs) is not None:
        for file in files :
            test.append(float(file.split('_')[0].split('-')[-1]))
            
            
toast = [j-i for i, j in zip(test[:-1], test[1:])]
a = 1/np.array(toast)
for i in toast:
    print(1/i)
    
plt.plot(a)
 