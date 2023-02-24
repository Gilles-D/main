# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:48:20 2023

@author: Gilles.DELBECQ
"""

import os
import sys
import pandas as pd
from matplotlib import pyplot as plt 
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy import stats
import statistics as stat
import math
import seaborn as sns

"""
------------------------------PARAMETERS------------------------------
"""
#Sessions idexes to analyse (as a list)
sessions_to_analyze = [1,3]


#Load MOCAP class
sys.path.append(r'D:\Gilles.DELBECQ\GitHub\main\MOCAP\Analysis')
import MOCAP_analysis_class as MA


#Directories location
root_dir='//equipe2-nas1/Gilles.DELBECQ/Data/MOCAP/Cohorte 2/CSV_gap_filled'
flat_csv_path='//equipe2-nas1/Gilles.DELBECQ/Data/MOCAP/Cohorte 2/CSV_gap_filled_flat'


data_info_path = 'D:/Gilles.DELBECQ/GitHub/main/MOCAP/Data/Session_Data_Cohorte2.xlsx'
data_info = MA.DATA_file(data_info_path)


savefig_path=r'\\equipe2-nas1\Gilles.DELBECQ\Data\MOCAP\Cohorte 2\Figs\Height_steps_med_gap_filled/'


saving_extension="png"


"""
------------------------------FUNCTIONS------------------------------
"""


def phase_correlation(t1, t2):
     """
     Calcule la phase relative entre deux trajectoires en 3D, dans les
trois dimensions.

     Args:
         t1 (ndarray): Un tableau de forme (n_frames, 3) représentant la
première trajectoire.
         t2 (ndarray): Un tableau de forme (n_frames, 3) représentant la
deuxième trajectoire.

     Returns:
         phase (ndarray): Un tableau de forme (3,) contenant les phases
relatives entre les deux trajectoires, dans les trois dimensions.
     """
     import numpy as np
     # Calcule la corrélation croisée entre les deux trajectoires dans chaque dimension
     c = [np.correlate(t1[:, i], t2[:, i], mode='same') for i in range(3)]

     # Détermine l'indice où la corrélation croisée est maximale dans chaque dimension
     peak_idx = [np.argmax(np.abs(c[i])) for i in range(3)]

     # Calcule la phase relative entre les deux trajectoires dans chaque dimension
     
     n_frames = t1.shape[0]
     phase = [(peak_idx[i] - n_frames//2) % n_frames for i in range(3)]

     return np.array(phase)
 
    
 
def phase_correlation_all_frames(window_size, traj1, traj2):
     n_frames = traj1.shape[0]

     phase_all_frames = np.zeros((n_frames - window_size, 3))

     for i in range(n_frames - window_size):
         window1 = traj1[i:i+window_size]
         window2 = traj2[i:i+window_size]

         phase = phase_correlation(window1, window2)

         phase_all_frames[i] = phase

     phase_all_frames = phase_all_frames[window_size//2:window_size//2]

     return phase_all_frames
 

"""
------------------------------FILE LOADING------------------------------
"""
#First Loop : loop on all csv files to list them in the list "Files"
Files = []
for r, d, f in os.walk(root_dir):
# r=root, d=directories, f = files
    for filename in f:
        if '.csv' in filename:
            Files.append(os.path.join(r, filename))
            
print('Files to analyze : {}'.format(len(Files)))
i=1


"""
------------------------------MAIN LOOP------------------------------
"""

for file in Files:
    #Load csv file
    data_MOCAP = MA.MOCAP_file(file)
    
    #Get Trial, session idexes
    idx = data_MOCAP.whole_idx()
    
    traj1=np.array(data_MOCAP.coord(f"{data_MOCAP.subject()}:Left_Foot")).transpose()
    traj2=np.array(data_MOCAP.coord(f"{data_MOCAP.subject()}:Left_Foot")).transpose()
  

    test = phase_correlation_all_frames(100, traj1,traj2)
    print(test)
