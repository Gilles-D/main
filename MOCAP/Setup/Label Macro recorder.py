# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 10:39:02 2023

@author: MOCAP
"""


import serial
import time
from datetime import datetime
import pyautogui
import numpy as np

from pynput import keyboard


pos_list=[]

def on_press(key):
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
    except AttributeError:
        print('special key {0} pressed'.format(
            key))
    return key
        

# ...or, in a non-blocking fashion:
listener = keyboard.Listener(
    on_press=on_press)
listener.start()

