# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 14:28:30 2022

@author: Gil
"""

import time
import cv2
import mss
import numpy
import read_grad


with mss.mss() as sct:
    
    monitor = {"top" : 90, "left" : 1600, "width" : 70, "height" : 60}   
    
    while "Screen capturing":
        last_time = time.time()
        
        # raw_img = sct.grab(monitor)
        img = numpy.array(sct.grab(monitor))
        
        output = "sct-{top}x{left}_{width}x{height}.png".format(**monitor)
        # mss.tools.to_png(raw_img.rgb, raw_img.size, output=output)
        
        img = img[:,:,:3]
        grad = read_grad.read(img)
        
        if len(grad) > 0 and grad[-1] == '%':
            print(grad[:-1])
            gradient=int(grad[:-1])
            
        time.sleep(2)