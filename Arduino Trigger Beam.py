# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 14:15:03 2020

@author: MOCAP
"""


import serial
import time
from datetime import datetime
import pyautogui

ser = serial.Serial('COM4', baudrate = 9600, timeout=0.5)
Timestamps=[]

"""Open softwares and positions windows"""


"""IR Beam triggers WinWCP"""
print("Waiting for the trigger")
while 1:
    arduinoData = ser.readline().decode('ascii')
    timestamp = datetime.fromtimestamp(time.time())
    #timestamp = str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    if arduinoData == "Broken\r\n":
        Timestamps.append(timestamp)
        print("Triggered. Start recording")
        pyautogui.click(1380, 118) #Click record on WinWCP
        time.sleep(3)#Sleep for 60 sec
    if arduinoData == "Arm\r\n":
        print("Arming")
        pyautogui.click(74, 40) #Click record on WinWCP
        time.sleep(2)#Sleep for 60 sec