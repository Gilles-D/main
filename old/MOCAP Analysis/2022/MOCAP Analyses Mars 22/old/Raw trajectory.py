# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 11:15:08 2022

@author: Gilles.DELBECQ
"""


"""
Fig2 : Raw trajectory

To do :
    IR Beam
"""
import os
import sys
import pandas as pd
from matplotlib import pyplot as plt 
import numpy as np
from scipy.signal import savgol_filter
import statistics as stat
import math

sys.path.append(r'C:\Users\Gilles.DELBECQ\Desktop\Python\MOCAP')
import MOCAP_analysis_class_v7 as MA


root_dir='D:\Working_Dir\MOCAP\Fev2022\Raw_CSV'

data_info_path = 'D:/Working_Dir/MOCAP/Fev2022/Data_info.xlsx'
data_info = MA.DATA_file(data_info_path)

savefig_path='D:\Working_Dir\MOCAP\Fev2022\Figs\Raw_traj'

#First Loop : loop on all csv files to list them in the list "Files"
Files = []
for r, d, f in os.walk(root_dir):
# r=root, d=directories, f = files
    for filename in f:
        if '.csv' in filename:
            Files.append(os.path.join(r, filename))
            
print('Files to analyze : {}'.format(len(Files)))

i=1

for file in Files:
    data_MOCAP = MA.MOCAP_file(file)
    print(file)
    idx = data_MOCAP.whole_idx()
    left_foot=data_MOCAP.coord(f"{data_MOCAP.subject()}:Left_Foot")
    right_foot=data_MOCAP.coord(f"{data_MOCAP.subject()}:Right_Foot")
    

    
    figure2 = plt.figure()
    plt.title(f'Raw Feet trajectory {data_MOCAP.subject()}_{data_MOCAP.session_idx()}_{data_MOCAP.trial_idx()}')
    plt.plot(-left_foot[1],left_foot[2],color='red',label='Left')
    plt.plot(-right_foot[1],right_foot[2],color='blue',label='Right')
    plt.axvline(stat.median(data_MOCAP.coord(f"{data_MOCAP.subject()}:IR Beam1")[1]))

    plt.savefig(f"{savefig_path}/{idx[0]}_{idx[1]}_{idx[2]}_raw.png")
    # plt.savefig(f"{savefig_path}/{idx[0]}_{idx[1]}_{idx[2]}_raw.svg")
    plt.close('all')
    
    left_foot=data_MOCAP.flatten(f"{data_MOCAP.subject()}:Left_Foot")
    right_foot=data_MOCAP.flatten(f"{data_MOCAP.subject()}:Right_Foot")
    
    figure2 = plt.figure()
    plt.title(f'Raw Feet trajectory flatten {data_MOCAP.subject()}_{data_MOCAP.session_idx()}_{data_MOCAP.trial_idx()}')
    plt.plot(-left_foot[0],left_foot[2],color='red',label='Left')
    plt.plot(-right_foot[0],right_foot[2],color='blue',label='Right')
    plt.axvline(stat.median(data_MOCAP.coord(f"{data_MOCAP.subject()}:IR Beam1")[1]))

    plt.savefig(f"{savefig_path}/{idx[0]}_{idx[1]}_{idx[2]}.png")
    # plt.savefig(f"{savefig_path}/{idx[0]}_{idx[1]}_{idx[2]}.svg")
    plt.close('all')    
    
    
    print(f"{i}/{len(Files)}")
    i=i+1
