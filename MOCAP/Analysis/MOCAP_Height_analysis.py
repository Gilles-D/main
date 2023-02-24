# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 15:54:49 2021

@author: Gilles.DELBECQ

Height of stance analysis

To do :
    separer baseline trial et before


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


Save_Peak_Detection = True
Save_Box_Plots = True

saving_extension="png"



"""
------------------------------FUNCTIONS------------------------------
"""


def flatten(t):
    return [item for sublist in t for item in sublist]

def peaks(x,y,z):
    peaks=find_peaks(z,prominence=5)[0]

    return peaks






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



#Get animals list
animals=[]

for file in Files:
    animals.append(file.split('\\')[-1].split('_')[0])

animals = list(dict.fromkeys(animals))






"""
------------------------------MAIN LOOP------------------------------
"""

for animal in animals:
    Height_list_left,Height_list_right=[],[]
    print(animal)
    for file in Files:
        data_MOCAP = MA.MOCAP_file(file)
        
        if data_MOCAP.subject() == animal:
            if int(data_MOCAP.session_idx()) in sessions_to_analyze: #Takes only first 2 sessions (basic locomotion)
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
                                 flat_coords[f"{idx[0]}:Left_Foot_Z"].tolist())
                right_peaks=peaks(flat_coords[f"{idx[0]}:Right_Foot_X"].tolist(),
                                  flat_coords[f"{idx[0]}:Right_Foot_Y"].tolist(),
                                  flat_coords[f"{idx[0]}:Right_Foot_Z"].tolist())

                
                
                
                
                """
                ------------------------------Plot peak detection------------------------------
                """
                
                #Plot the trajectory
                figure = plt.figure()
                plt.plot(flat_coords[f"{idx[0]}:Left_Foot_X"].tolist(),
                         flat_coords[f"{idx[0]}:Left_Foot_Z"].tolist(),c='red')
                plt.plot(flat_coords[f"{idx[0]}:Right_Foot_X"].tolist(),
                         flat_coords[f"{idx[0]}:Right_Foot_Z"].tolist(),c='blue')

                plt.gca().invert_xaxis()
                

                #Empty lists to Categorize peaks as before or after le IR Beam                
                height_before_left,height_after_left=[],[]
                height_before_right,height_after_right=[],[]
                
                #Left foot
                for i in left_peaks:
                    if info[0] != "Off": #If it is a stim on trial
                        plt.title(f'Height step detection {data_MOCAP.subject()}_{data_MOCAP.session_idx()}_{data_MOCAP.trial_idx()} Stim On')
                        
                        if flat_coords[f"{idx[0]}:Left_Foot_X"][i] >=beam: #Before the beam
                            height_before_left.append(
                                flat_coords[f"{idx[0]}:Left_Foot_Z"][i])
                            plt.plot(
                                flat_coords[f"{idx[0]}:Left_Foot_X"][i], 
                                flat_coords[f"{idx[0]}:Left_Foot_Z"][i], "o",c='red')
                            
                        else: #After the beam
                            height_after_left.append(
                                flat_coords[f"{idx[0]}:Left_Foot_Z"][i])
                            plt.plot(flat_coords[f"{idx[0]}:Left_Foot_X"][i],
                                     flat_coords[f"{idx[0]}:Left_Foot_Z"][i], "o",c='pink')
                            
                    else: #It is not a stim trial
                        plt.title(f'Height step detection {data_MOCAP.subject()}_{data_MOCAP.session_idx()}_{data_MOCAP.trial_idx()} Stim Off')
                        height_before_left.append(flat_coords[f"{idx[0]}:Left_Foot_Z"][i])
                        plt.plot(flat_coords[f"{idx[0]}:Left_Foot_X"][i], 
                                 flat_coords[f"{idx[0]}:Left_Foot_Z"][i], "o",c='red')
                        
                
                #Find the median height for left and right steps to substract it later to each height before beam 
                med_left = stat.median(height_before_left)
                
                
                # Height_list_left.append([idx,height_before_left,height_after_left])
                Height_list_left.append([idx,height_before_left-med_left,height_after_left-med_left])
                
                
                #Right foot
                for i in right_peaks:
                    if info[0] != "Off":#If it is a stim on trial
                        plt.title(f'Height step detection {data_MOCAP.subject()}_{data_MOCAP.session_idx()}_{data_MOCAP.trial_idx()} Stim On')
                        plt.axvline(beam)
                        if flat_coords[f"{idx[0]}:Right_Foot_X"][i] >=beam:#Before the beam
                            height_before_right.append(
                                flat_coords[f"{idx[0]}:Right_Foot_Z"][i])
                            plt.plot(flat_coords[f"{idx[0]}:Right_Foot_X"][i], 
                                     flat_coords[f"{idx[0]}:Right_Foot_Z"][i], "o",c='blue')
                        else:#After the beam
                            height_after_right.append(
                                flat_coords[f"{idx[0]}:Right_Foot_Z"][i])
                            plt.plot(flat_coords[f"{idx[0]}:Right_Foot_X"][i], 
                                     flat_coords[f"{idx[0]}:Right_Foot_Z"][i], "o",c='cyan')
                    else:#It is not a stim trial
                        plt.title(f'Height step detection {data_MOCAP.subject()}_{data_MOCAP.session_idx()}_{data_MOCAP.trial_idx()} Stim Off')
                        height_before_right.append(flat_coords[f"{idx[0]}:Right_Foot_Z"][i])
                        plt.plot(flat_coords[f"{idx[0]}:Right_Foot_X"][i], 
                                 flat_coords[f"{idx[0]}:Right_Foot_Z"][i], "o",c='blue')
                
                #Find the median height for left and right steps to substract it later to each height before beam
                med_right = stat.median(height_before_right)
                
                # Height_list_right.append([idx,height_before_right,height_after_right])
                Height_list_right.append([idx,height_before_right-med_right,height_after_right-med_right])
                
                
                if Save_Peak_Detection == True:
                    MA.Check_Save_Dir(savefig_path)
                    plt.savefig(f"{savefig_path}/{idx[0]}_{idx[1]}_{idx[2]}_steps.{saving_extension}")
                    plt.close('all')



    """
    ------------------------------Plot Box plot for each side------------------------------
    """


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
    
    
    plt.title(f"Right {animal}")

    # Draw a nested boxplot to show bills by day and time
    sns.boxplot(data=df_right)
    # sns.swarmplot(data=df_right,color='black')
    # plt.ylim(-6, 15)
    
    
    figure2=plt.figure()
    plt.title(f"Left {animal}")
    
    # Draw a nested boxplot to show bills by day and time
    sns.boxplot(data=df_left)
    # sns.swarmplot(data=df_left,color='black')
    # plt.ylim(-6, 15)
    
    if Save_Peak_Detection == True:
        MA.Check_Save_Dir(savefig_path)
        figure1.savefig(f"{savefig_path}/Right_steps_{animal}.{saving_extension}")
        figure2.savefig(f"{savefig_path}/Left_steps_{animal}.{saving_extension}")