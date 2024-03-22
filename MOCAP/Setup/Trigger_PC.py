# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:22:07 2021

@author: MOCAP

WINWCP :
    - Record : x=1475, y=117
    - protocol select  x=1481, y=279
    - 
"""


import serial
import time
from datetime import datetime
import pyautogui
import numpy as np

ser = serial.Serial('COM4', baudrate = 9600, timeout=0.5)
test=[]
Timestamps=[]

Record_status = 0

print("Waiting for the trigger")
while 1:
    arduinoData = ser.readline().decode('ascii')
    print(arduinoData)
    test.append(arduinoData)
    if str(arduinoData) == 'Start\r\n' or str(arduinoData) == 16753245:
        print("Start Recording")
        pyautogui.click(2388, 692) #Click start Vicon
        Timestamps.append(datetime.fromtimestamp(time.time()))


    if str(arduinoData) == 'Stop\r\n':
        print("Stop Recording")
        pyautogui.click(2387, 692) #Click stop Vicon
        # pyautogui.click(1888, -111)
        # time.sleep(0.1)#Sleep  
        # pyautogui.click(1888, -111) #Click stop AMP
        # time.sleep(0.5)#Sleep    
        # pyautogui.click(1888, -111) #Click reset AMP

    
    elif arduinoData == "Broken\r\n":
        print("Broken. Start Stimulating")
        # pyautogui.click(1475, 118) #Click record on WinWCP
        # time.sleep(3)#Sleep for 60 sec


"""
    if str(arduinoData) == 'FFE21D\r\n':
        print('randomize protocol')
        pyautogui.click(1481, 279)
        pyautogui.click(1481, 279)
        pyautogui.press('up')
        pyautogui.press('up')
        pyautogui.press('up')
        pyautogui.press('up')
        pyautogui.press('up')
        a = np.random.randint(1, 5, 1)
        if a == 1:
            pyautogui.press('down')
            pyautogui.click(1153, 319)
            pyautogui.press('backspace')
            pyautogui.press('backspace')
            pyautogui.write('20')
            
        if a == 2:
            pyautogui.press('down')
            pyautogui.press('down')
            pyautogui.click(1153, 319)
            pyautogui.press('backspace')
            pyautogui.press('backspace')
            pyautogui.write('40')
        if a == 3:
            pyautogui.press('down')
            pyautogui.press('down')
            pyautogui.press('down')
            pyautogui.click(1153, 319)
            pyautogui.press('backspace')
            pyautogui.press('backspace')
            pyautogui.write('60')
        if a == 4:
            pyautogui.press('down')
            pyautogui.press('down')
            pyautogui.press('down')
            pyautogui.press('down')
            pyautogui.click(1153, 319)
            pyautogui.press('backspace')
            pyautogui.press('backspace')
            pyautogui.write('80')
"""