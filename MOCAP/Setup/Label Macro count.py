# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 16:48:20 2023

@author: MOCAP
"""

import pyautogui
from pynput import keyboard, mouse

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

click_count = 0  # Variable pour compter les clics de souris
scroll_count = 0  # Variable pour compter les coups de molette
key_press_count = 0  # Variable pour compter les touches de clavier pressées

def on_press(key):
    global key_press_count
    try:
        print('alphanumeric key {0} pressed'.format(key.char))

        for i in range(len(key_list)):
            if key.char == key_list[i]:
                current_pos = pyautogui.position()
                pyautogui.click(pos_list[i][0], pos_list[i][1])
                pyautogui.moveTo(current_pos[0], current_pos[1])

        key_press_count += 1  # Incrémenter le compteur de touches de clavier pressées
        print('Touches de clavier pressées :', key_press_count)

    except AttributeError:
        print('special key {0} pressed'.format(key))

def on_click(x, y, button, pressed):
    global click_count, scroll_count
    try:
        print('{0} at {1}'.format('toast', button))
        if pressed:
            if str(button) == 'Button.x2':
                pyautogui.press('right')
            if str(button) == 'Button.x1':
                pyautogui.press('left')

            click_count += 1  # Incrémenter le compteur de clics de souris
            print('Clics de souris :', click_count)
    except:
        print("except")

def on_scroll(x, y, dx, dy):
    global scroll_count
    scroll_count += 1  # Incrémenter le compteur de coups de molette
    print('Coups de molette :', scroll_count)

listener1 = mouse.Listener(on_click=on_click, on_scroll=on_scroll)
listener1.start()

listener = keyboard.Listener(on_press=on_press)
listener.start()

try:
    listener1.join()
    listener.join()
except KeyboardInterrupt:
    pass
