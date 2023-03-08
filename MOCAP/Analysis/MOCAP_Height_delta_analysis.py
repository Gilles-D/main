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


Save_Peak_Detection = False
Save_Box_Plots = False

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
    Delta_off_left,Delta_on_left,Delta_off_right,Delta_on_right=[],[],[],[]
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
                left_peaks=peaks(flat_coords[f"{idx[0]}:Left_Ankle_X"].tolist(),
                                 flat_coords[f"{idx[0]}:Left_Ankle_Y"].tolist(),
                                 flat_coords[f"{idx[0]}:Left_Ankle_Z"].tolist())
                right_peaks=peaks(flat_coords[f"{idx[0]}:Right_Ankle_X"].tolist(),
                                  flat_coords[f"{idx[0]}:Right_Ankle_Y"].tolist(),
                                  flat_coords[f"{idx[0]}:Right_Ankle_Z"].tolist())

                
                left_lows=peaks(flat_coords[f"{idx[0]}:Left_Ankle_X"].tolist(),
                                 flat_coords[f"{idx[0]}:Left_Ankle_Y"].tolist(),
                                 -flat_coords[f"{idx[0]}:Left_Ankle_Z"])
                right_lows=peaks(flat_coords[f"{idx[0]}:Right_Ankle_X"].tolist(),
                                  flat_coords[f"{idx[0]}:Right_Ankle_Y"].tolist(),
                                  -flat_coords[f"{idx[0]}:Right_Ankle_Z"])                
                
                             
                
                
                left_delta,right_delta=[],[]
                
                
                
                if len(left_peaks) > 1:#Si pas vide
                    if left_lows[0] < left_peaks[1]:
                        for i in range(len(left_lows)):
                            try:
                                delta=flat_coords[f"{idx[0]}:Left_Ankle_Z"][left_peaks[i]]-flat_coords[f"{idx[0]}:Left_Ankle_Z"][left_lows[i]]
                                if info[0] == "Off":
                                    Delta_off_left.append(delta)
                                else:
                                    if flat_coords[f"{idx[0]}:Left_Ankle_X"][left_lows[i]]<beam:
                                        Delta_on_left.append(delta)
                            except:
                                pass
                    else:
                        for i in range(len(left_lows)):
                            delta=flat_coords[f"{idx[0]}:Left_Ankle_Z"][left_peaks[i+1]]-flat_coords[f"{idx[0]}:Left_Ankle_Z"][left_lows[i]]
                            
                            for i in range(len(left_lows)):
                                try:
                                    delta=flat_coords[f"{idx[0]}:Left_Ankle_Z"][left_peaks[i+1]]-flat_coords[f"{idx[0]}:Left_Ankle_Z"][left_lows[i]]
                                    
                                    if info[0] == "Off":
                                        Delta_off_left.append(delta)
                                    else:
                                        if flat_coords[f"{idx[0]}:Left_Ankle_X"][left_lows[i]]<beam:
                                            Delta_on_left.append(delta)
                                except:
                                    pass
                
                if len(right_peaks) > 1:#Si pas vide
                    if right_lows[0] < right_peaks[1]:
                        for i in range(len(right_lows)):
                            try:
                                delta=flat_coords[f"{idx[0]}:Right_Ankle_Z"][right_peaks[i]]-flat_coords[f"{idx[0]}:Right_Ankle_Z"][right_lows[i]]
                                if info[0] == "Off":
                                    Delta_off_right.append(delta)
                                else:
                                    if flat_coords[f"{idx[0]}:Right_Ankle_X"][right_lows[i]]<beam:
                                        Delta_on_right.append(delta)
                            except:
                                pass
                    else:
                        for i in range(len(right_lows)):
                            delta=flat_coords[f"{idx[0]}:Right_Ankle_Z"][right_peaks[i+1]]-flat_coords[f"{idx[0]}:Right_Ankle_Z"][right_lows[i]]
                            
                            for i in range(len(right_lows)):
                                try:
                                    delta=flat_coords[f"{idx[0]}:Right_Ankle_Z"][right_peaks[i+1]]-flat_coords[f"{idx[0]}:Right_Ankle_Z"][right_lows[i]]
                                    
                                    if info[0] == "Off":
                                        Delta_off_right.append(delta)
                                    else:
                                        if flat_coords[f"{idx[0]}:Right_Ankle_X"][right_lows[i]]<beam:
                                            Delta_on_right.append(delta)
                                except:
                                    pass                   
                
                

                
                """
                ------------------------------Plot peak detection------------------------------
                """
                
                # #Plot the trajectory
                # figure = plt.figure()
                # plt.plot(flat_coords[f"{idx[0]}:Left_Ankle_X"].tolist(),
                #          flat_coords[f"{idx[0]}:Left_Ankle_Z"].tolist(),c='red')
                # # plt.plot(flat_coords[f"{idx[0]}:Right_Ankle_X"].tolist(),
                # #          flat_coords[f"{idx[0]}:Right_Ankle_Z"].tolist(),c='blue')

                # plt.gca().invert_xaxis()
                # for i in left_peaks:
                #     plt.plot(
                #         flat_coords[f"{idx[0]}:Left_Ankle_X"][i], 
                #         flat_coords[f"{idx[0]}:Left_Ankle_Z"][i], "o",c='red')
                
                # for i in left_lows:
                #     plt.plot(
                #         flat_coords[f"{idx[0]}:Left_Ankle_X"][i], 
                #         flat_coords[f"{idx[0]}:Left_Ankle_Z"][i], "o",c='red')
                
                # plt.plot(left_delta)
        
        
        
        
                if Save_Peak_Detection == True:
                    MA.Check_Save_Dir(savefig_path)
                    plt.savefig(f"{savefig_path}/{idx[0]}_{idx[1]}_{idx[2]}_steps.{saving_extension}")
                    plt.close('all')



    """
    ------------------------------Plot Box plot for each side------------------------------
    """


    sns.set_theme(style="ticks", palette="pastel")
    figure1=plt.figure()
    
    df = pd.DataFrame({'Left Off':pd.Series(Delta_off_left),'Left On':pd.Series(Delta_on_left),'Right Off':pd.Series(Delta_off_right),'Right On':pd.Series(Delta_on_right)})

    plt.title(f"Right {animal}")

    # Draw a nested boxplot to show bills by day and time
    sns.violinplot(data=df)
    # sns.swarmplot(data=df_right,color='black')
    # plt.ylim(-6, 15)
    

    if Save_Peak_Detection == True:
        MA.Check_Save_Dir(savefig_path)
        figure1.savefig(f"{savefig_path}/Right_steps_{animal}.{saving_extension}")
        figure2.savefig(f"{savefig_path}/Left_steps_{animal}.{saving_extension}")