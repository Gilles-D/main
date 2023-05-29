# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:47:33 2020

@author: MOCAP
"""


import cv2
import numpy as np
import glob
import os
import re
import shutil

rootdir = r'D:\SOD_2023\_7_24_05'

img_array = []


for subdirs, dirs, files in os.walk(rootdir):
    start_index = []
    folder = 1
    for file in files:
        result = file.endswith('_1_0.bmp')
        if result == True:
            start_index.append(files.index(file))
    
    while folder <= len(start_index):
        if folder == 1:
            if not os.path.exists("{}/{}".format(subdirs, folder)):
                os.makedirs("{}/{}".format(subdirs, folder))
            print(folder)    
            for file in files[0:start_index[folder]]:
               shutil.move("{}/{}".format(subdirs,file), "{}/{}/{}".format(subdirs, folder,file))
            folder+=1
        
        else:

              
            if not folder == len(start_index):
                print(folder)
                if not os.path.exists("{}/{}".format(subdirs, folder)):
                    os.makedirs("{}/{}".format(subdirs, folder))
                for file in files[start_index[folder-1]:start_index[folder]]:
                    shutil.move("{}/{}".format(subdirs,file), "{}/{}/{}".format(subdirs, folder,file))
                folder+=1
                
            if folder == len(start_index):
                print(folder)
                if not os.path.exists("{}/{}".format(subdirs, folder)):
                    os.makedirs("{}/{}".format(subdirs, folder))
                for file in files[start_index[folder-1]:]:
                    shutil.move("{}/{}".format(subdirs,file), "{}/{}/{}".format(subdirs, folder,file))
                folder+=1