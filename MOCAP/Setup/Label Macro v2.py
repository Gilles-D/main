# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 10:38:24 2023

@author: MOCAP
"""

import pyautogui
import threading
import time

from pynput import keyboard
from pynput import mouse

pyautogui.FAILSAFE = False

key_list = ['a', 'z', 'q', 's', 'd', 'f', 'w', 'x', 'c', 'v']

pos_list = [
    (2220, 288),
    (2220, 306),
    (2220, 324),
    (2220, 342),
    (2220, 360),
    (2220, 378),
    (2220, 396),
    (2220, 414),
    (2220, 432),
    (2220, 450)
]

scrolling = False

def scroll_wheel():
    global scrolling
    while scrolling:
        pyautogui.scroll(100)  # Emulate wheel scroll up
        time.sleep(0.001)  # 500 ms interval

def on_press(key):
    global scrolling
    try:
        if key.char != '<':
            print('alphanumeric key {0} pressed'.format(key.char))
            
        for i in range(len(key_list)):
            if key.char == key_list[i]:
                current_pos = pyautogui.position()
                pyautogui.click(pos_list[i][0], pos_list[i][1])
                pyautogui.moveTo(current_pos[0], current_pos[1])
        
        # Start scrolling when '<' key is pressed
        if key.char == '<':
            if not scrolling:
                scrolling = True
                threading.Thread(target=scroll_wheel).start()

    except AttributeError:
        print('special key {0} pressed'.format(key))

def on_release(key):
    global scrolling
    try:
        # Stop scrolling when '<' key is released
        if key.char == '<':
            scrolling = False
    except AttributeError:
        pass

def on_click(x, y, button, pressed):
    try:
        print('{0} at {1}'.format('toast', button))
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
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
