# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 13:29:01 2022

@author: Gilles.DELBECQ
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
save_dir='D:\Working_Dir\MOCAP\Fev2022\Flat_CSV'

data_info_path = 'D:/Working_Dir/MOCAP/Fev2022/Data_info.xlsx'
data_info = MA.DATA_file(data_info_path)

#First Loop : loop on all csv files to list them in the list "Files"
Files = []
for r, d, f in os.walk(root_dir):
# r=root, d=directories, f = files
    for filename in f:
        if '.csv' in filename:
            Files.append(os.path.join(r, filename))
            
print('Files to analyze : {}'.format(len(Files)))


for file in Files:
    print(file)
    data_MOCAP = MA.MOCAP_file(file)
    
    left_foot=data_MOCAP.coord(f"{data_MOCAP.subject()}:Left_Foot")
    right_foot=data_MOCAP.coord(f"{data_MOCAP.subject()}:Right_Foot")
    

    
    figure2 = plt.figure()
    plt.title(f'Raw Feet trajectory {data_MOCAP.subject()}_{data_MOCAP.session_idx()}_{data_MOCAP.trial_idx()}')
    plt.plot(-left_foot[1],left_foot[2],color='red',label='Left')
    plt.plot(-right_foot[1],right_foot[2],color='blue',label='Right')
    plt.axvline(stat.median(data_MOCAP.coord(f"{data_MOCAP.subject()}:IR Beam1")[1]))
    
    start = data_MOCAP.coord(f"{data_MOCAP.subject()}:Platform1")
    stop =  data_MOCAP.coord(f"{data_MOCAP.subject()}:Platform2")
    start_x,start_z=stat.median(start[1]),stat.median(start[2])
    stop_x,stop_z=stat.median(stop[1]),stat.median(stop[2])     
    
    plt.plot(-start_x,start_z,"o",c='pink')
    plt.plot(-stop_x,stop_z,"o",c='pink')
    
    plt.plot([-start_x,-stop_x],[start_z+20,stop_z+20])
