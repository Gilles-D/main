# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:29:05 2023

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


savefig_path=r'\\equipe2-nas1\Gilles.DELBECQ\Data\MOCAP\Cohorte 2\Figs\Normalized_trajectory/'

Save = True
saving_extension="png"

Flat = True

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
    
    #Get coords for each foot
    left_foot=data_MOCAP.normalized(f"{data_MOCAP.subject()}:Left_Hip", f"{data_MOCAP.subject()}:Left_Foot")
    right_foot=data_MOCAP.normalized(f"{data_MOCAP.subject()}:Right_Hip", f"{data_MOCAP.subject()}:Right_Foot")
    
    
    #Plot trajectory of each foot
    figure1 = plt.figure()
    plt.title(f'Normalized Feet trajectory {data_MOCAP.subject()}_{data_MOCAP.session_idx()}_{data_MOCAP.trial_idx()}')
    plt.plot(-left_foot[1],left_foot[2],color='red',label='Left')
    plt.plot(-right_foot[1],right_foot[2],color='blue',label='Right')

    
    if Save == True:
        MA.Check_Save_Dir(savefig_path)
        plt.savefig(f"{savefig_path}/{idx[0]}_{idx[1]}_{idx[2]}.{saving_extension}")
        plt.close('all')
    
    print(f"{i}/{len(Files)}")
    i=i+1
