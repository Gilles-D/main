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

savefig_path='D:\Working_Dir\MOCAP\Fev2022\Figs\Height_steps'

#First Loop : loop on all csv files to list them in the list "Files"
Files = []
for r, d, f in os.walk(root_dir):
# r=root, d=directories, f = files
    for filename in f:
        if '.csv' in filename:
            Files.append(os.path.join(r, filename))
            
print('Files to analyze : {}'.format(len(Files)))


def peaks(x,y,z):
    peaks=find_peaks(z,prominence=5)[0]

    return peaks

Height_list_left,Height_list_right=[],[]


for file in Files:
    data_MOCAP = MA.MOCAP_file(file)
    
    if int(data_MOCAP.session_idx()) in [1,2]: #Takes only first 2 sessions (basic locomotion)
        print(file)
        
        idx = data_MOCAP.whole_idx()
        info = data_info.get_info(data_MOCAP.subject(),data_MOCAP.session_idx(),
                                  data_MOCAP.trial_idx())
        
        beam = stat.median(data_MOCAP.coord(f"{data_MOCAP.subject()}:IR Beam1")[1])
        
        flat_coords=pd.read_csv(f"{flat_csv_path}/{idx[0]}_{idx[1]}_{idx[2]}.csv")

        
        left_peaks=peaks(flat_coords[f"{idx[0]}:Left_Foot_X"].tolist(),
                         flat_coords[f"{idx[0]}:Left_Foot_Y"].tolist(),
                         flat_coords[f"{idx[0]}:Left_Foot_Z"].tolist())
        right_peaks=peaks(flat_coords[f"{idx[0]}:Right_Foot_X"].tolist(),
                          flat_coords[f"{idx[0]}:Right_Foot_Y"].tolist(),
                          flat_coords[f"{idx[0]}:Right_Foot_Z"].tolist())
        
        figure = plt.figure()
        plt.plot(flat_coords[f"{idx[0]}:Left_Foot_X"].tolist(),
                 flat_coords[f"{idx[0]}:Left_Foot_Z"].tolist(),c='red')
        plt.plot(flat_coords[f"{idx[0]}:Right_Foot_X"].tolist(),
                 flat_coords[f"{idx[0]}:Right_Foot_Z"].tolist(),c='blue')
        
        

        
        plt.gca().invert_xaxis()
        
        height_before_left,height_after_left=[],[]
        
        for i in left_peaks:
            if math.isnan(info[0]) == False:
                plt.title(f'Height step detection {data_MOCAP.subject()}_{data_MOCAP.session_idx()}_{data_MOCAP.trial_idx()} Stim On')
                if flat_coords[f"{idx[0]}:Left_Foot_X"][i] >=beam:
                    height_before_left.append(
                        flat_coords[f"{idx[0]}:Left_Foot_Z"][i])
                    plt.plot(
                        flat_coords[f"{idx[0]}:Left_Foot_X"][i], 
                        flat_coords[f"{idx[0]}:Left_Foot_Z"][i], "o",c='red')
                else:

                    height_after_left.append(
                        flat_coords[f"{idx[0]}:Left_Foot_Z"][i])
                    plt.plot(flat_coords[f"{idx[0]}:Left_Foot_X"][i],
                             flat_coords[f"{idx[0]}:Left_Foot_Z"][i], "o",c='pink')
            else:
                plt.title(f'Height step detection {data_MOCAP.subject()}_{data_MOCAP.session_idx()}_{data_MOCAP.trial_idx()} Stim Off')
                height_before_left.append(flat_coords[f"{idx[0]}:Left_Foot_Z"][i])
                plt.plot(flat_coords[f"{idx[0]}:Left_Foot_X"][i], 
                         flat_coords[f"{idx[0]}:Left_Foot_Z"][i], "o",c='red')
        
        Height_list_left.append([idx,height_before_left,height_after_left])
        
        height_before_right,height_after_right=[],[]
        
        for i in right_peaks:
            if math.isnan(info[0]) == False:
                plt.title(f'Height step detection {data_MOCAP.subject()}_{data_MOCAP.session_idx()}_{data_MOCAP.trial_idx()} Stim On')
                plt.axvline(beam)
                if flat_coords[f"{idx[0]}:Right_Foot_X"][i] >=beam:
                    height_before_right.append(
                        flat_coords[f"{idx[0]}:Right_Foot_Z"][i])
                    plt.plot(flat_coords[f"{idx[0]}:Right_Foot_X"][i], 
                             flat_coords[f"{idx[0]}:Right_Foot_Z"][i], "o",c='blue')
                else:
                    height_after_right.append(
                        flat_coords[f"{idx[0]}:Right_Foot_Z"][i])
                    plt.plot(flat_coords[f"{idx[0]}:Right_Foot_X"][i], 
                             flat_coords[f"{idx[0]}:Right_Foot_Z"][i], "o",c='cyan')
            else:
                plt.title(f'Height step detection {data_MOCAP.subject()}_{data_MOCAP.session_idx()}_{data_MOCAP.trial_idx()} Stim Off')
                height_before_right.append(flat_coords[f"{idx[0]}:Right_Foot_Z"][i])
                plt.plot(flat_coords[f"{idx[0]}:Right_Foot_X"][i], 
                         flat_coords[f"{idx[0]}:Right_Foot_Z"][i], "o",c='blue')
        
        Height_list_right.append([idx,height_before_right,height_after_right])
        
        plt.savefig(f"{savefig_path}/{idx[0]}_{idx[1]}_{idx[2]}_steps.png")
        plt.savefig(f"{savefig_path}/{idx[0]}_{idx[1]}_{idx[2]}_steps.svg")
        plt.close('all')
        


sns.set_theme(style="ticks", palette="pastel")
figure1=plt.figure()
r_b,r_a,l_b,l_a=[],[],[],[]


for i in Height_list_right:
    r_b.append(i[1])
    r_a.append(i[2])

for i in Height_list_left:
    l_b.append(i[1])
    l_a.append(i[2])

r_b = flatten(r_b)
r_a = flatten(r_a)
l_b = flatten(l_b)
l_a = flatten(l_a)

df = pd.DataFrame({'Before':r_b})
df2 = pd.DataFrame({'After':r_a})
df3 = pd.DataFrame({'Before':l_b})
df4 = pd.DataFrame({'After':l_a})

df_right = pd.concat([df, df2], axis=1) 
df_left = pd.concat([df3, df4], axis=1) 


plt.title("Right")




# Draw a nested boxplot to show bills by day and time
sns.boxplot(data=df_right)
sns.swarmplot(data=df_right,color='black')
plt.ylim(10, 40)


figure2=plt.figure()
plt.title("Left")

# Draw a nested boxplot to show bills by day and time
sns.boxplot(data=df_left)
sns.swarmplot(data=df_left,color='black')
plt.ylim(10, 40)


figure1.savefig(f"{savefig_path}/Right_steps.svg")
figure1.savefig(f"{savefig_path}/Right_steps.png")
figure2.savefig(f"{savefig_path}/Left_steps.svg")
figure2.savefig(f"{savefig_path}/Left_steps.png")