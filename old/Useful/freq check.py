# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:21:08 2023

@author: MOCAP
"""
import cv2
import numpy as np
import glob
import os
import re



rootdir = 'D:/DLC/Capture/05_cam1_100%_01/6'


test=[]

for subdirs, dirs, files in os.walk(rootdir):
    if re.search(r'\d+$', subdirs) is not None:
        for file in files :
            test.append(float(file.split('_')[0].split('-')[-1]))
            
            
toast = [j-i for i, j in zip(test[:-1], test[1:])]
for i in toast:
    print(1/i)