# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 15:54:49 2021

@author: Gilles.DELBECQ

length of stance analysis


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
sys.path.append(r'C:\Users\Gilles.DELBECQ\Desktop\Python\MOCAP')
import MOCAP_analysis_class_v7 as MA

def flatten(t):
    return [item for sublist in t for item in sublist]

root_dir='D:\Working_Dir\MOCAP\Fev2022\Raw_CSV'

data_info_path = 'D:/Working_Dir/MOCAP/Fev2022/Data_info.xlsx'
data_info = MA.DATA_file(data_info_path)

flat_csv_path='D:\Working_Dir\MOCAP\Fev2022\Flat_CSV'

savefig_path='D:\Working_Dir\MOCAP\Fev2022\Figs\length_steps_med'

#First Loop : loop on all csv files to list them in the list "Files"
Files = []
for r, d, f in os.walk(root_dir):
# r=root, d=directories, f = files
    for filename in f:
        if '.csv' in filename:
            Files.append(os.path.join(r, filename))
            
print('Files to analyze : {}'.format(len(Files)))

animals=[]

for file in Files:
    animals.append(file.split('\\')[-1].split('_')[0])

animals = list(dict.fromkeys(animals))

def peaks(x,y,z):
    peaks=find_peaks(-z,prominence=5)[0]

    return peaks


for animal in animals:
    X_list_left,X_list_right=[],[]
    print(animal)
    for file in Files:
        data_MOCAP = MA.MOCAP_file(file)
        if data_MOCAP.subject() == animal:
            if int(data_MOCAP.session_idx()) in [1,2]: #Takes only first 2 sessions (basic locomotion)
                print(file)
                
                #Takes file info
                idx = data_MOCAP.whole_idx()
                info = data_info.get_info(data_MOCAP.subject(),data_MOCAP.session_idx(),
                                          data_MOCAP.trial_idx())
                
                #Get beam coord
                beam = stat.median(data_MOCAP.coord(f"{data_MOCAP.subject()}:IR Beam1")[1])
                
                flat_coords=MA.Flat_CSV(f"{flat_csv_path}/{idx[0]}_{idx[1]}_{idx[2]}.csv")
                df_flat=flat_coords.dataframe()
                
                # Get cycle frame idx
                left_cycles = flat_coords.get_cycles('Left_Foot')
                right_cycles = flat_coords.get_cycles('Right_Foot')
                
                
