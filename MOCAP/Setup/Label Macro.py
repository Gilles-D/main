# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 10:38:24 2023

@author: MOCAP
"""

import pyautogui

from pynput import keyboard
from pynput import mouse


pyautogui.FAILSAFE = False


key_list=['a','z','q','s','d','f','w','x','c','v']

pos_list=[
    (2220,288),
    (2220,306),
    (2220,324),
    (2220,342),
    (2220,360),
    (2220,378),
    (2220,396),
    (2220,414),
    (2220,432),
    (2220,450)
    
    ]

def on_press(key):
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
        
        for i in range(len(key_list)):
            if key.char == key_list[i]:
                current_pos=pyautogui.position()
                pyautogui.click(pos_list[i][0],pos_list[i][1])
                pyautogui.moveTo(current_pos[0],current_pos[1])
    except AttributeError:
        print('special key {0} pressed'.format(
            key))


def on_click(x, y, button, pressed):
    try :
        print('{0} at {1}'.format('toast',button))
        if pressed:
            if str(button) == 'Button.x2':
                pyautogui.press('right')
            if str(button) == 'Button.x1':
                pyautogui.press('left')
            
    except:
        print("except")            

# ...or, in a non-blocking fashion:
listener1 = mouse.Listener(on_click=on_click)
listener1.start()

# ...or, in a non-blocking fashion:
listener = keyboard.Listener(
    on_press=on_press)
listener.start()

# # Collect events until released
# with keyboard.Listener(
#         on_press=on_press) as listener:
#     listener.join()
    
