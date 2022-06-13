# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 15:25:58 2020

@author: MOCAP
"""



from PIL import Image, ImageOps
import os
import re
rootdir = r'C:/Users/MOCAP/Pictures/IC Express/Ladder 1234 Session 1-200Hz_'
savedir=r'D:/Ladder Limb 2309/'


for subdirs, dirs, files in os.walk(rootdir):
    for file in files:
        file_path = os.path.join(subdirs, file)
        img = Image.open(file_path)
        img_mirror = ImageOps.mirror(img)
        savedir_file = r'{}\{}\{}'.format(savedir,re.split(r'/', re.sub(r'\\', '/', subdirs))[-2],re.split(r'/', re.sub(r'\\', '/', subdirs))[-1])
        if not os.path.exists(savedir_file):
            os.makedirs(savedir_file)
            
        img_mirror.save( r'{}\{}.jpg'.format(savedir_file,file))