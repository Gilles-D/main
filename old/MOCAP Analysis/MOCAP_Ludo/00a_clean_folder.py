# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 17:39:26 2020

Removes .c3d, .history, .system and other non-csv files from a given folder

@author: Ludovic.SPAETH
"""

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

sup_folder = 'U:/10_MOCAP/Thy1Beam_CORRECTED/D2'

import os 

for session in ['{}/{}'.format(sup_folder,x) for x in os.listdir(sup_folder) if 'Patient' not in x]:
    
    print (session)
    
    for file in ['{}/{}'.format(session,x) for x in os.listdir(session)] :
        
        if file.endswith('.csv'):
            print('{} is conserved'.format(file.split('/')[-1]))
            continue
        
        else:
            os.remove(file)
            