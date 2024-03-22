# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 10:38:24 2023

@author: MOCAP
"""

import pyautogui
from pynput import keyboard
from pynput import mouse
from time import sleep  # Importe la fonction sleep depuis le module time pour gérer les délais

pyautogui.FAILSAFE = False

# Liste des touches à presser
keys = ['left', 'right', 'right', 'right', 'delete', '1','enter']





def on_press(key):
    try:
        # Vérifie si la touche pressée est 'z'
        if key.char == 'z':
            # Sauvegarde la position actuelle de la souris
            current_pos = pyautogui.position()
            
            # Clique droit à la position actuelle de la souris
            pyautogui.click(button='right')
            sleep(0.5)
            # Déplace la souris à la position (X+82, Y+35) par rapport à sa position actuelle
            new_pos = (current_pos[0] + 82, current_pos[1] + 35)
            pyautogui.moveTo(new_pos)
            sleep(0.2)
            z
            # Clique gauche à la nouvelle position
            pyautogui.click()
            sleep(0.2)
            # Simule les pressions sur les touches fléchées et la touche Supprimer
            # Boucle à travers chaque touche de la liste
            for key_to_press in keys:
                pyautogui.press(key_to_press)  # Presse la touche
                sleep(0.03)  # Attend 200ms (0.2 secondes) avant de presser la prochaine touche


    except AttributeError:
        # Gère les touches spéciales si nécessaire
        pass

# Initialise les écouteurs pour le clavier et la souris
listener = keyboard.Listener(on_press=on_press)
listener.start()

mouse_listener = mouse.Listener()
mouse_listener.start()

# Le script continue de s'exécuter en arrière-plan, en attendant une pression de touche
