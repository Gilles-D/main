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
 


def make_video(files):
    img_array = []
    for file in files:
        file_path = os.path.join(subdirs, file) 
        print(file_path)
        img = cv2.imread(file_path)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    a = re.sub(r'\\', '/', subdirs)
    a = re.split(r'/', a)
    a = "%s %s" % (a[5], a[6])
    a.replace(' ', '_')
    print('Saving...')
    out = cv2.VideoWriter('{}{}.avi'.format(rootdir, a),cv2.VideoWriter_fourcc(*'DIVX'), 100, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    return(out.release())

rootdir = 'C:/Users/MOCAP/Pictures/IC Express/'

img_array = []

for subdirs, dirs, files in os.walk(rootdir):
    if re.search(r'\d+$', subdirs) is not None:
        make_video(files)
 