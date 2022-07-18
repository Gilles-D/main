# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 10:34:23 2022

@author: Gil
"""

import glob
import cv2
import numpy as np
from skimage.metrics import structural_similarity
import time
import mss

from datetime import datetime
best = [None,0]

while True:
    
    """
    Screen capture
    """
    
    with mss.mss() as sct:
        monitor = {"top" : 70, "left" : 1810, "width" : 90, "height" : 60}   
    
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        
        img = np.array(sct.grab(monitor)) 
        img = img[:,:,0]
        
    
    """
    Compare
    """
    
    for file in glob.glob('G:\Github\main\other\**\*.png', recursive=True):
        img_ref = cv2.imread(file,0)
        grad = file.split('\\')[-2]
        val = structural_similarity(img, img_ref)
        
        if val > best[1]:
            best = [grad,val]
        
    print(best)
    
    time.sleep(0.5)

