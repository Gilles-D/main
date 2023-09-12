# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 08:48:20 2023

@author: Gilles Delbecq

Extracts data from MOCAP output xslx files
Uses MOCAP class

Output : excel files for analysis (positions of bodyparts, angles, distances, stances)

"""


import os
import sys
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt 

#Load MOCAP class
sys.path.append(r'C:\Users\MOCAP\Documents\GitHub\main\MOCAP\Analysis')
import MOCAP_analysis_class as MA

#%%Functions

def list_all_files(root):
    #First Loop : loop on all csv files to list them in the list "Files"
    Files = []
    for r, d, f in os.walk(root):
    # r=root, d=directories, f = files
        for filename in f:
            if '.csv' in filename:
                Files.append(os.path.join(r, filename))
                
    print('Files to analyze : {}'.format(len(Files)))
    
    return Files


def file_info(file):
    filename = os.path.basename(file)
    animal_name = int(filename.split('_')[0])
    session = int(filename.split('_')[1])
    trial = int(filename.split('_')[-1].split('.')[0])
    
    file_info = {
    "filename": filename,
    "animal_name": animal_name,
    "session": session,
    "trial": trial
    }
    
    return file_info
    


#%% Parameters
#Directory location of raw csvs
mocap_data_folder=r'D:\ePhy\SI_Data\mocap_files\Auto-comp'

force_rewrite = True

Stance = True
control_plot = True

#%% Parameters computation

#Export the data : 
# relative positions foot - ankle - knee - hip from back Left and Right
# angles ankle - knee - hip Left and right
# speed
# position from obstacle
# position back1 from start
# position  back1 from end


Files = list_all_files(mocap_data_folder)

for i,file in enumerate(Files):
    file_infos = file_info(file)
    #Load csv file
    data_MOCAP = MA.MOCAP_file(file)
    
    #Get Trial, session idexes
    idx = data_MOCAP.whole_idx()
    
    save_folder = rf"{os.path.dirname(file)}/Analysis/"
    save_path = rf"{save_folder}/Analysis_{idx[0]}_{idx[1]}_{idx[2]}.xlsx"
    
    if os.path.isfile(save_path) and force_rewrite == False:
        print(rf"{save_path} already exists")
        
    else :
        #Get normalized coords for side
        left_foot_norm=data_MOCAP.normalized(f"{data_MOCAP.subject()}:Back1", f"{data_MOCAP.subject()}:Left_Foot")
        left_ankle_norm = data_MOCAP.normalized(f"{data_MOCAP.subject()}:Back1", f"{data_MOCAP.subject()}:Left_Ankle")
        left_knee_norm = data_MOCAP.normalized(f"{data_MOCAP.subject()}:Back1", f"{data_MOCAP.subject()}:Left_Knee")
        left_hip_norm = data_MOCAP.normalized(f"{data_MOCAP.subject()}:Back1", f"{data_MOCAP.subject()}:Left_Hip")
        
        right_foot_norm=data_MOCAP.normalized(f"{data_MOCAP.subject()}:Back1", f"{data_MOCAP.subject()}:Right_Foot")
        right_ankle_norm = data_MOCAP.normalized(f"{data_MOCAP.subject()}:Back1", f"{data_MOCAP.subject()}:Right_Ankle")
        right_knee_norm = data_MOCAP.normalized(f"{data_MOCAP.subject()}:Back1", f"{data_MOCAP.subject()}:Right_Knee")
        right_hip_norm = data_MOCAP.normalized(f"{data_MOCAP.subject()}:Back1", f"{data_MOCAP.subject()}:Right_Hip")
        
        
        #Get coord for obstacle
        obstacle = data_MOCAP.coord(f"{data_MOCAP.subject()}:Obstacle1")
        
        #Compute the angles
        
        left_ankle_angle = data_MOCAP.calculate_angle(f"{data_MOCAP.subject()}:Left_Foot", f"{data_MOCAP.subject()}:Left_Ankle", f"{data_MOCAP.subject()}:Left_Knee")
        left_knee_angle = data_MOCAP.calculate_angle(f"{data_MOCAP.subject()}:Left_Ankle", f"{data_MOCAP.subject()}:Left_Knee", f"{data_MOCAP.subject()}:Left_Hip")
        left_hip_angle = data_MOCAP.calculate_angle(f"{data_MOCAP.subject()}:Left_Knee", f"{data_MOCAP.subject()}:Left_Hip", f"{data_MOCAP.subject()}:Back1")
        
        right_ankle_angle = data_MOCAP.calculate_angle(f"{data_MOCAP.subject()}:Right_Foot", f"{data_MOCAP.subject()}:Right_Ankle", f"{data_MOCAP.subject()}:Right_Knee")
        right_knee_angle = data_MOCAP.calculate_angle(f"{data_MOCAP.subject()}:Right_Ankle", f"{data_MOCAP.subject()}:Right_Knee", f"{data_MOCAP.subject()}:Right_Hip")
        right_hip_angle = data_MOCAP.calculate_angle(f"{data_MOCAP.subject()}:Right_Knee", f"{data_MOCAP.subject()}:Right_Hip", f"{data_MOCAP.subject()}:Back1")
        
        back1 = data_MOCAP.coord(f"{data_MOCAP.subject()}:Back1")
        back2 = data_MOCAP.coord(f"{data_MOCAP.subject()}:Back2")
        back_orientation = np.degrees(np.arctan((back2[2] - back1[2]) / np.sqrt((back2[1] - back1[1])**2 + (back2[0] - back1[0])**2)))
        back_inclination = back2[2] - back1[2]
        
        back_1_Z = back1[2]
        back_2_Z = back2[2]
        
        
        # Compute Speed
        
        speed_back1 = np.insert(data_MOCAP.speed(f"{data_MOCAP.subject()}:Back1"),0,np.nan)
        speed_left_foot = np.insert(data_MOCAP.speed(f"{data_MOCAP.subject()}:Left_Foot"),0,np.nan)
        speed_right_foot = np.insert(data_MOCAP.speed(f"{data_MOCAP.subject()}:Right_Foot"),0,np.nan)
           
        # Distance from obstacle   
        distance_from_obstacle = data_MOCAP.distance_from(f"{data_MOCAP.subject()}:Back1", f"{data_MOCAP.subject()}:Obstacle1")
        
        # Distance from start
        distance_from_start =  data_MOCAP.distance_from(f"{data_MOCAP.subject()}:Back1", f"{data_MOCAP.subject()}:Platform1")
    
        # Distance from end
        distance_from_end = data_MOCAP.distance_from(f"{data_MOCAP.subject()}:Back1", f"{data_MOCAP.subject()}:Platform2")
    
        
        #Setup a array with all parameters
        
        data = {
            
            
            
            'left_foot_x_norm' : left_foot_norm[1],
            'left_foot_y_norm' : left_foot_norm[0],
            'left_foot_z_norm' : left_foot_norm[2],
            
            'left_ankle_x_norm' : left_ankle_norm[1],
            'left_ankle_y_norm' : left_ankle_norm[0],
            'left_ankle_z_norm' : left_ankle_norm[2],
            
            'left_knee_x_norm' : left_knee_norm[1],
            'left_knee_y_norm' : left_knee_norm[0],
            'left_knee_z_norm' : left_knee_norm[2],
            
            'left_hip_x_norm' : left_hip_norm[1],
            'left_hip_y_norm' : left_hip_norm[0],
            'left_hip_z_norm' : left_hip_norm[2],
            
            'right_foot_x_norm' : right_foot_norm[1],
            'right_foot_y_norm' : right_foot_norm[0],
            'right_foot_z_norm' : right_foot_norm[2],
            
            'right_ankle_x_norm' : right_ankle_norm[1],
            'right_ankle_y_norm' : right_ankle_norm[0],
            'right_ankle_z_norm' : right_ankle_norm[2],
            
            'right_knee_x_norm' : right_knee_norm[1],
            'right_knee_y_norm' : right_knee_norm[0],
            'right_knee_z_norm' : right_knee_norm[2],
            
            'right_hip_x_norm' : right_hip_norm[1],
            'right_hip_y_norm' : right_hip_norm[0],
            'right_hip_z_norm' : right_hip_norm[2],
            
            'left_ankle_angle' : left_ankle_angle,
            'left_knee_angle' : left_knee_angle,
            'left_hip_angle' : left_hip_angle,
            
            'right_ankle_angle' : right_ankle_angle,
            'right_knee_angle' : right_knee_angle,
            'right_hip_angle' : right_hip_angle,
            
            'back1_x' : back1[1],
            'back1_y' : back1[0],
            'back1_z' : back1[2],
            
            'back2_x' : back2[1],
            'back2_y' : back2[0],
            'back2_z' : back2[2],
            
            'back_orientation' : back_orientation,
            'back_inclination' : back_inclination,
            'back_1_Z' : back_1_Z,
            'back_2_Z' : back_2_Z,
            
            'speed_back1' : speed_back1,
            'speed_left_foot' : speed_left_foot,
            'speed_right_foot' : speed_right_foot,
            
            'obstacle_x' : obstacle[1],
            'obstacle_y' : obstacle[0],
            'obstacle_z' : obstacle[2],
            
            'distance_from_obstacle' : distance_from_obstacle,
            'distance_from_start' : distance_from_start,
            'distance_from_end' : distance_from_end,
            
            }
        
        df = pd.DataFrame(data)
        
        
        #Save as csv
        
        MA.Check_Save_Dir(save_folder)
        
        df.to_excel(save_path)
        
    
        if control_plot == True:
            #Plot trajectory of each foot
            figure1 = plt.figure()
            plt.title(f'Normalized Feet trajectory {data_MOCAP.subject()}_{data_MOCAP.session_idx()}_{data_MOCAP.trial_idx()}')
            plt.plot(-left_foot_norm[1],left_foot_norm[2],color='red',label='Left')
            plt.plot(-right_foot_norm[1],right_foot_norm[2],color='blue',label='Right')
        
    
        
        print(f"{i+1}/{len(Files)}")
        
    if Stance == True:
        save_path_stance = rf"{save_folder}/Stances_{idx[0]}_{idx[1]}_{idx[2]}.xlsx"
        
        if os.path.isfile(save_path_stance):
            print(rf"{save_path_stance} already exists")
        else:
            stance_left = data_MOCAP.stance(f"{data_MOCAP.subject()}:Left_Foot")
            stance_right = data_MOCAP.stance(f"{data_MOCAP.subject()}:Right_Foot")
            
            data_stance = {
                "stance_left_idx" : stance_left[0],
                "stance_left_x" : stance_left[1],
                "stance_left_y" : stance_left[2],
                "stance_left_z" : stance_left[3],
                "stance_left_lengths" : np.insert(stance_left[4],0,np.nan),

                "stance_right_idx" : stance_left[0],
                "stance_right_x" : stance_left[1],
                "stance_right_y" : stance_left[2],
                "stance_right_z" : stance_left[3],
                "stance_right_lengths" : np.insert(stance_left[4],0,np.nan),                

                }
            
            df_stance = pd.DataFrame(data_stance)
            
            df_stance.to_excel(save_path_stance)
                
    
                