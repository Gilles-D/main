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
    
    #Détecter pics bas
    #prendre les 2 premiers
    #calculer coef dir
    #flatten
    
    
    """
    Use class function to flatten the coords for both feet using the 2 reference points on the platform
    """
    left_foot_flat = np.transpose(data_MOCAP.flatten(f"{data_MOCAP.subject()}:Left_Foot"))
    right_foot_flat = np.transpose(data_MOCAP.flatten(f"{data_MOCAP.subject()}:Right_Foot"))
    
    """
    Saves it as a dataframe
    """
    data=np.concatenate((left_foot_flat,right_foot_flat),axis=1)
    df = pd.DataFrame(data)
    names=[f"{data_MOCAP.subject()}:Left_Foot_X",f"{data_MOCAP.subject()}:Left_Foot_Y",f"{data_MOCAP.subject()}:Left_Foot_Z",
           f"{data_MOCAP.subject()}:Right_Foot_X",f"{data_MOCAP.subject()}:Right_Foot_Y",f"{data_MOCAP.subject()}:Right_Foot_Z"]
    df.columns = names
    df.to_csv(rf"{save_dir}\{data_MOCAP.subject()}_{data_MOCAP.session_idx()}_{data_MOCAP.trial_idx()}.csv") 