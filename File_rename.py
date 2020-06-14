# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:41:51 2020

@author: MOCAP
"""


from PIL import Image
import os

rootdir = r'C:\Users\MOCAP\Pictures\IC Express\Cam_Right'
1
i=1

for subdirs, dirs, files in os.walk(rootdir):
    for file in files:
        file_path = os.path.join(subdirs, file) 
        img = Image.open(file_path)
        img.save( r'{}\Cam_Right_{}.jpg'.format(rootdir,i))
        i+=1

rootdir = r'C:\Users\MOCAP\Pictures\IC Express\Cam_Left'

i=1

for subdirs, dirs, files in os.walk(rootdir):
    for file in files:
        file_path = os.path.join(subdirs, file) 
        img = Image.open(file_path)
        img.save( r'{}\Cam_Left_{}.jpg'.format(rootdir,i))
        i+=1