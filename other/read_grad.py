# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 14:35:01 2022

@author: Gil
"""

import glob
import cv2
import numpy as np
from skimage.metrics import structural_similarity

def crop_image(img):
    non_blank_rows = [i for i in range(img.shape[0]) if sum(img[i,:]) != 0]
    img = img[non_blank_rows[0]:,:]
    img = img[:non_blank_rows[-1]-non_blank_rows[0],:]
    non_blank_cols = [i for i in range(img.shape[1]) if sum(img[i,:]) != 0]
    img = img[non_blank_cols[0]:,:]
    img = img[:non_blank_cols[-1]-non_blank_cols[0],:]
    
    return img

def match_img(img):
    best = [None,0]
    for file in glob.glob('ref_chars/*.png'):
        val = structural_similarity(img, cv2.imread(file,0))
        if val > best[1]:
            best = [file,val]
    return best[0].split('/')[-1].split('.')[0]

def read(img):
    mask1 = cv2.inRange(img, (10,190,240),(20,200,250))
    mask2 = cv2.inRange(img, (60,100,240),(70,110,250))
    mask1 = cv2.bitwise_or(mask1,mask2)
    mask3 = cv2.inRange(img, (0,0,250),(0,0,255))
    mask1 = cv2.bitwise_or(mask1,mask3)
    mask3 = cv2.inRange(img, (250,250,250),(255,255,255))
    mask = cv2.bitwise_or(mask1,mask3)
    
    blank_cols = [i for i in range(mask.shape[1]-1) if sum(mask[:,i]) == 0 and sum(mask[:,i+1]) != 0]
    blank_cols.append(mask.shape[1])
    
    string = ''
    
    for i in range(len(blank_cols) -1):
        charector = mask[:,blank_cols[i]:blank_cols[i+1]]
        charector = crop_image(charector)
        charector = cv2.resize(charector, (99,99))
        string += match_img(charector)
        
        return string