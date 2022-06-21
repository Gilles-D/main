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
                
                #Get flattened coords
                flat_coords=pd.read_csv(f"{flat_csv_path}/{idx[0]}_{idx[1]}_{idx[2]}.csv")
        
                #Detect peaks
                left_peaks=peaks(flat_coords[f"{idx[0]}:Left_Foot_X"].tolist(),
                                 flat_coords[f"{idx[0]}:Left_Foot_Y"].tolist(),
                                 flat_coords[f"{idx[0]}:Left_Foot_Z"])
                right_peaks=peaks(flat_coords[f"{idx[0]}:Right_Foot_X"].tolist(),
                                  flat_coords[f"{idx[0]}:Right_Foot_Y"].tolist(),
                                  flat_coords[f"{idx[0]}:Right_Foot_Z"])
                
                
                
                
                
                
                #Plot peaks detection figure
                figure = plt.figure()
                plt.plot(flat_coords[f"{idx[0]}:Left_Foot_X"].tolist(),
                         flat_coords[f"{idx[0]}:Left_Foot_Z"].tolist(),c='red')
                plt.plot(flat_coords[f"{idx[0]}:Right_Foot_X"].tolist(),
                         flat_coords[f"{idx[0]}:Right_Foot_Z"].tolist(),c='blue')

                plt.gca().invert_xaxis()
                
                
                
                #Categorize peaks as before or after le IR Beam                
                x_before_left,x_after_left=[],[]
                x_before_right,x_after_right=[],[]
                
                #Left foot
                t=0
                for i in left_peaks:
                    if math.isnan(info[0]) == False: #If it is a stim on trial
                        plt.title(f'length step detection {data_MOCAP.subject()}_{data_MOCAP.session_idx()}_{data_MOCAP.trial_idx()} Stim On')
                        
                        if flat_coords[f"{idx[0]}:Left_Foot_X"][i] >=beam: #Before the beam
                            x_before_left.append(
                                flat_coords[f"{idx[0]}:Left_Foot_X"][i])
                            plt.plot(
                                flat_coords[f"{idx[0]}:Left_Foot_X"][i], 
                                flat_coords[f"{idx[0]}:Left_Foot_Z"][i], "o",c='red')
                            
                        else: #After the beam
                            x_after_left.append(
                                flat_coords[f"{idx[0]}:Left_Foot_X"][i])
                            plt.plot(flat_coords[f"{idx[0]}:Left_Foot_X"][i],
                                     flat_coords[f"{idx[0]}:Left_Foot_Z"][i], "o",c='pink')
                            
                    else: #It is not a stim trial
                        plt.title(f'length step detection {data_MOCAP.subject()}_{data_MOCAP.session_idx()}_{data_MOCAP.trial_idx()} Stim Off')
                        x_before_left.append(flat_coords[f"{idx[0]}:Left_Foot_X"][i])
                        plt.plot(flat_coords[f"{idx[0]}:Left_Foot_X"][i], 
                                 flat_coords[f"{idx[0]}:Left_Foot_Z"][i], "o",c='red')
                    t=t+1        
                
                X_list_left.append([idx,x_before_left,x_after_left])
                
                
                #Right foot
                t=0
                for i in right_peaks:
                    if math.isnan(info[0]) == False:#If it is a stim on trial
                        plt.title(f'length step detection {data_MOCAP.subject()}_{data_MOCAP.session_idx()}_{data_MOCAP.trial_idx()} Stim On')
                        plt.axvline(beam)
                        if flat_coords[f"{idx[0]}:Right_Foot_X"][i] >=beam:#Before the beam
                            x_before_right.append(
                                flat_coords[f"{idx[0]}:Right_Foot_X"][i])
                            plt.plot(flat_coords[f"{idx[0]}:Right_Foot_X"][i], 
                                     flat_coords[f"{idx[0]}:Right_Foot_Z"][i], "o",c='blue')
                        else:#After the beam
                            x_after_right.append(
                                flat_coords[f"{idx[0]}:Right_Foot_X"][i])
                            plt.plot(flat_coords[f"{idx[0]}:Right_Foot_X"][i], 
                                     flat_coords[f"{idx[0]}:Right_Foot_Z"][i], "o",c='cyan')
                    else:#It is not a stim trial
                        plt.title(f'length step detection {data_MOCAP.subject()}_{data_MOCAP.session_idx()}_{data_MOCAP.trial_idx()} Stim Off')
                        x_before_right.append(flat_coords[f"{idx[0]}:Right_Foot_X"][i])
                        plt.plot(flat_coords[f"{idx[0]}:Right_Foot_X"][i], 
                                 flat_coords[f"{idx[0]}:Right_Foot_Z"][i], "o",c='blue')
                    t=t+1
                
                X_list_right.append([idx,x_before_right,x_after_right])
                
                plt.savefig(f"{savefig_path}/{idx[0]}_{idx[1]}_{idx[2]}_steps.png")
                plt.savefig(f"{savefig_path}/{idx[0]}_{idx[1]}_{idx[2]}_steps.svg")
                plt.close('all')

    sns.set_theme(style="ticks", palette="pastel")
    figure1=plt.figure()
    r_b,r_a,l_b,l_a=[],[],[],[]
    
    
    for i in X_list_right:
        r_b.append(i[1])
        r_a.append(i[2])
    
    Length_right_before=[]
    for i in r_b:
        for j in range(len(i)):
            if j != 0:
                Length_right_before.append(i[j]-i[j-1])
    
    Length_right_after=[]
    for i in r_a:
        for j in range(len(i)):
            if j != 0:
                Length_right_after.append(i[j]-i[j-1])    
    
    
    for i in X_list_left:
        l_b.append(i[1])
        l_a.append(i[2])
    
    Length_left_before=[]
    for i in l_b:
        for j in range(len(i)):
            if j != 0:
                Length_left_before.append(i[j]-i[j-1])
    
    Length_left_after=[]
    for i in l_a:
        for j in range(len(i)):
            if j != 0:
                Length_left_after.append(i[j]-i[j-1]) 


    
    df = pd.DataFrame({'Before':Length_right_before})
    df2 = pd.DataFrame({'After':Length_right_after})
    df3 = pd.DataFrame({'Before':Length_left_before})
    df4 = pd.DataFrame({'After':Length_left_after})
    
    df_right = pd.concat([df, df2], axis=1) 
    df_left = pd.concat([df3, df4], axis=1) 
    
    
    plt.title(f"Right {animal}")

    # Draw a nested boxplot to show bills by day and time
    sns.boxplot(data=df_right)
    sns.swarmplot(data=df_right,color='black')
    # plt.ylim(-3, 3)
    
    
    figure2=plt.figure()
    plt.title(f"Left {animal}")
    
    # Draw a nested boxplot to show bills by day and time
    sns.boxplot(data=df_left)
    sns.swarmplot(data=df_left,color='black')
    # plt.ylim(-3, 3)
    
    
    
    figure1.savefig(f"{savefig_path}/Right_steps_{animal}.svg")
    figure1.savefig(f"{savefig_path}/Right_steps_{animal}.png")
    figure2.savefig(f"{savefig_path}/Left_steps_{animal}.svg")
    figure2.savefig(f"{savefig_path}/Left_steps_{animal}.png")