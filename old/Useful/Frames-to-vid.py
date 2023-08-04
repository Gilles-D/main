# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 14:28:03 2020

@author: Gil
"""

import cv2
import numpy as np
import glob
import os
import re
 


def make_video(files,freq):
    img_array = []
    for i,file in enumerate(files):
        print(rf'{i+1} / {len(files)}')
        file_path = os.path.join(subdirs, file) 
        # print(file_path)
        img = cv2.imread(file_path)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    name = re.sub(r'\\', '/', subdirs).split(r'/')
    name = rf'{name[-2]}_{name[-1]}'
    print('Saving...')
    print('{}/{}.avi'.format(os.path.dirname(rootdir), name))
    out = cv2.VideoWriter('{}/{}.avi'.format(os.path.dirname(rootdir), name),cv2.VideoWriter_fourcc(*'DIVX'), freq, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    return(out.release())

def check_freq(files):
    timestamps=[]
    for file in files :
        timestamps.append(float(file.split('_')[0].split('-')[-1])) 
    time_delays = [j-i for i, j in zip(timestamps[:-1], timestamps[1:])]
    freq = 1/np.array(time_delays)
    
    print(np.average(freq))
    
    return(np.average(freq))


list_root_dir=[
"D:/Videos/0012/0023_31_07/1",
"D:/Videos/0012/0023_31_07/2",
"D:/Videos/0012/0023_31_07/3",
"D:/Videos/0012/0023_31_07/4",
"D:/Videos/0012/XXXX_02_08/1",
"D:/Videos/0012/XXXX_02_08/2",
"D:/Videos/0012/XXXX_01_08/1",
"D:/Videos/0012/XXXX_01_08/2",
"D:/Videos/0012/XXXX_01_08/3",
"D:/Videos/0012/XXXX_01_08/4",
"D:/Videos/0012/XXXX_01_08/5",
"D:/Videos/0012/0026_01_08/5",
"D:/Videos/0012/0026_01_08/4",
"D:/Videos/0012/0026_01_08/3",
"D:/Videos/0012/0026_01_08/2",
"D:/Videos/0012/0026_01_08/1",
"D:/Videos/0012/0025_29_07/1",
"D:/Videos/0012/0024_28_07/1",
"D:/Videos/0012/0024_03_08/1",
"D:/Videos/0012/0024_03_08/2",
"D:/Videos/0012/0023_31_07/1",
"D:/Videos/0012/0023_31_07/2",
"D:/Videos/0012/0023_31_07/3",
"D:/Videos/0012/0023_31_07/4"
 ]
# rootdir = r'D:/Videos/0012/0012_08_06/4'

img_array = []


for rootdir in list_root_dir:
    for subdirs, dirs, files in os.walk(rootdir):
        if re.search(r'\d+$', subdirs) is not None:
            freq= check_freq(files)
            make_video(files,freq)
     