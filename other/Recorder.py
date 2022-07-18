# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 17:11:30 2022

@author: Gil
"""

import time
import cv2
import mss
import numpy
import read_grad
from datetime import datetime




with mss.mss() as sct:
    
    monitor = {"top" : 70, "left" : 1810, "width" : 90, "height" : 60}   
    
    for i in range(3600):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        
        raw_img = sct.grab(monitor)
        output = f"char/screen_{now.strftime('%H_%M_%S')}.png".format(**monitor)
        mss.tools.to_png(raw_img.rgb, raw_img.size, output=output)
        
        time.sleep(2)