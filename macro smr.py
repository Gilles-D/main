# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 15:39:43 2021

@author: Gilles.DELBECQ
"""


import pyautogui
import keyboard
import time

List=[(-1905, 33), (-1833, 190), (-1043, 724), (-1036, 758)]

while True:  # making a loop
    try:  # used try so that if user pressed other than the given key error will not be shown
        if keyboard.is_pressed('x'):  # if key 'q' is pressed
            print('You Pressed A Key!')
            List.append((pyautogui.position()[0],pyautogui.position()[1]))
            time.sleep(1)
        if keyboard.is_pressed('z'):  # if key 'q' is pressed
            print('You Pressed The Key!')
            for i in range(len(List)):
                print((List[i][0], List[i][1]))
                pyautogui.click(List[i][0], List[i][1])
                time.sleep(0.5)
            time.sleep(1)
    except:
        break  # if user pressed a key other than the given key the loop will break