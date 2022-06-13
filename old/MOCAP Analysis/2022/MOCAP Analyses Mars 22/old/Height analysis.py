# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 15:54:49 2021

@author: Gilles.DELBECQ

Height of stance analysis


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

def flatten(t):
    return [item for sublist in t for item in sublist]

file_path = 'D:/Working_Dir/MOCAP/Fev2022/Raw_CSV/1110/1110_01_02.csv'
data_MOCAP = MA.MOCAP_file(file_path)




root_dir='D:\Working_Dir\MOCAP\Fev2022\Raw_CSV'

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


infos,peak_list,beam_list = [],[],[]

Height_list=[]

for file in Files:
    data_MOCAP = MA.MOCAP_file(file)
    if int(data_MOCAP.session_idx()) in [1,2]: #Takes only first 2 sessions (basic locomotion)
        print(file)

        idx = data_MOCAP.whole_idx()
        info = data_info.get_info(data_MOCAP.subject(),data_MOCAP.session_idx(),data_MOCAP.trial_idx())
        
        steps = data_MOCAP.step_height(f"{data_MOCAP.subject()}:Right_Foot")
        beam = stat.median(data_MOCAP.coord(f"{data_MOCAP.subject()}:IR Beam1")[1])
        height_before,height_after=[],[]
        for step in steps:
            if math.isnan(info[0]) == False:
                if step[1]>=beam:
                    height_before.append(step[3])
                else:
                    height_after.append(step[3])
                
            else:
                height_before.append(step[3])
                
        #To normalize with the median height before
        median_before = stat.median(height_before)
        height_before_norm,height_after_norm=[],[]
        
        for i in height_before:
            height_before_norm.append(i-median_before)
        for i in height_after:
            height_after_norm.append(i-median_before)        
        Height_list.append([idx,height_before_norm,height_after_norm])



plt.figure()
test=[]
test2=[]
for i in Height_list:
    test.append(i[1])
    test2.append(i[2])
    
test = flatten(test)
test2 = flatten(test2)

df = pd.DataFrame({'Before':test})
df2 = pd.DataFrame({'After':test2})

newdf = pd.concat([df, df2], axis=1) 

import seaborn as sns
sns.set_theme(style="ticks", palette="pastel")

# Draw a nested boxplot to show bills by day and time
sns.boxplot(data=newdf)
sns.despine(offset=10, trim=True)