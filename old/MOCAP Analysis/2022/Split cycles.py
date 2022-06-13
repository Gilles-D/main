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
from scipy.signal import find_peaks 
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

savefig_path='D:\Working_Dir\MOCAP\Fev2022\Figs\length_steps'

#First Loop : loop on all csv files to list them in the list "Files"
Files = []
for r, d, f in os.walk(root_dir):
# r=root, d=directories, f = files
    for filename in f:
        if '.csv' in filename:
            Files.append(os.path.join(r, filename))
            
print('Files to analyze : {}'.format(len(Files)))


def peaks(x,y,z):
    peaks=find_peaks(-z,prominence=5)[0]

    return peaks

Length_list_left,Length_list_right=[],[]


for file in Files:
    data_MOCAP = MA.MOCAP_file(file)
    
    if int(data_MOCAP.session_idx()) in [1,2]: #Takes only first 2 sessions (basic locomotion)
        print(file)
        
        idx = data_MOCAP.whole_idx()
        info = data_info.get_info(data_MOCAP.subject(),data_MOCAP.session_idx(),data_MOCAP.trial_idx())
        
        beam = stat.median(data_MOCAP.coord(f"{data_MOCAP.subject()}:IR Beam1")[1])
        
        flat_coords=pd.read_csv(f"{flat_csv_path}/{idx[0]}_{idx[1]}_{idx[2]}.csv")
        left_foot_x=flat_coords[f"{idx[0]}:Left_Foot_X"].tolist()
        left_foot_z=flat_coords[f"{idx[0]}:Left_Foot_Z"].tolist()
        
        right_foot_x=flat_coords[f"{idx[0]}:Right_Foot_X"].tolist()
        right_foot_z=flat_coords[f"{idx[0]}:Right_Foot_Z"].tolist()
        
        left_peaks=peaks(flat_coords[f"{idx[0]}:Left_Foot_X"].tolist(),flat_coords[f"{idx[0]}:Left_Foot_Y"],flat_coords[f"{idx[0]}:Left_Foot_Z"])
        right_peaks=peaks(flat_coords[f"{idx[0]}:Right_Foot_X"].tolist(),flat_coords[f"{idx[0]}:Right_Foot_Y"],flat_coords[f"{idx[0]}:Right_Foot_Z"])
        
        # data = left_foot
        # sizes = np.diff(list(left_peaks))
        # it = iter(data)
        # cycles = [[next(it) for _ in range(size)] for size in sizes]    
        
        cycles_left,cycles_right=[],[]
        i = iter(left_foot_x)
        x = [[next(i) for _ in range(size)] for size in np.diff(list(left_peaks))]
        i = iter(left_foot_z)
        z = [[next(i) for _ in range(size)] for size in np.diff(list(left_peaks))]  
        cycles_left=list(zip(x,z))
        
        y=iter(right_foot_x)
        x = [[next(y) for _ in range(size)] for size in np.diff(list(right_peaks))]  
        y=iter(right_foot_z)
        z = [[next(y) for _ in range(size)] for size in np.diff(list(right_peaks))]  
        cycles_right=list(zip(x,z))
        
        figure = plt.figure()
        plt.title(f'split cycle {data_MOCAP.subject()}_{data_MOCAP.session_idx()}_{data_MOCAP.trial_idx()}')
        for cycle in cycles_right:
            plt.plot(cycle[0],cycle[1])
        
        
        
        
    
        
        
