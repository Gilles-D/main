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
sys.path.append(r'D:\Gilles.DELBECQ\GitHub\main\MOCAP\Analysis')
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
mocap_data_folder=r'//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/mocap_files\Auto-comp\0022'

force_rewrite = False

Analysis = True
Stance = False
State = False
control_plot = False

#%% Parameters computation

#Export the data : 
# relative positions foot - ankle - knee - hip from back Left and Right
# angles ankle - knee - hip Left and right
# speed
# position from obstacle
# position back1 from start
# position  back1 from end


Files = list_all_files(mocap_data_folder)

for j,file in enumerate(Files):
    file_infos = file_info(file)
    #Load csv file
    data_MOCAP = MA.MOCAP_file(file)
    
    #Get Trial, session idexes
    idx = data_MOCAP.whole_idx()
    
    save_folder = rf"{os.path.dirname(file)}/Analysis/"
    save_path = rf"{save_folder}/Analysis_{idx[0]}_{idx[1]}_{idx[2]}.xlsx"
    
    if Analysis == True:
            
        
        if os.path.isfile(save_path) and force_rewrite == False:
            print(rf"{save_path} already exists")
            
        else :
            left_foot = data_MOCAP.coord(f"{data_MOCAP.subject()}:Left_Foot")
            left_ankle = data_MOCAP.coord(f"{data_MOCAP.subject()}:Left_Ankle")
            left_knee = data_MOCAP.coord(f"{data_MOCAP.subject()}:Left_Knee")
            left_hip = data_MOCAP.coord(f"{data_MOCAP.subject()}:Left_Hip")
            
            right_foot = data_MOCAP.coord(f"{data_MOCAP.subject()}:Right_Foot")
            right_ankle = data_MOCAP.coord(f"{data_MOCAP.subject()}:Right_Ankle")
            right_knee = data_MOCAP.coord(f"{data_MOCAP.subject()}:Right_Knee")
            right_hip = data_MOCAP.coord(f"{data_MOCAP.subject()}:Right_Hip")        
            
            #Get normalized coords for side
            left_foot_norm=data_MOCAP.normalized(f"{data_MOCAP.subject()}:Back1", f"{data_MOCAP.subject()}:Left_Foot")
            left_ankle_norm = data_MOCAP.normalized(f"{data_MOCAP.subject()}:Back1", f"{data_MOCAP.subject()}:Left_Ankle")
            left_knee_norm = data_MOCAP.normalized(f"{data_MOCAP.subject()}:Back1", f"{data_MOCAP.subject()}:Left_Knee")
            left_hip_norm = data_MOCAP.normalized(f"{data_MOCAP.subject()}:Back1", f"{data_MOCAP.subject()}:Left_Hip")
            
            right_foot_norm=data_MOCAP.normalized(f"{data_MOCAP.subject()}:Back1", f"{data_MOCAP.subject()}:Right_Foot")
            right_ankle_norm = data_MOCAP.normalized(f"{data_MOCAP.subject()}:Back1", f"{data_MOCAP.subject()}:Right_Ankle")
            right_knee_norm = data_MOCAP.normalized(f"{data_MOCAP.subject()}:Back1", f"{data_MOCAP.subject()}:Right_Knee")
            right_hip_norm = data_MOCAP.normalized(f"{data_MOCAP.subject()}:Back1", f"{data_MOCAP.subject()}:Right_Hip")
            
            #Get normalized coords for side
            left_foot_norm_hip=data_MOCAP.normalized(f"{data_MOCAP.subject()}:Left_Hip", f"{data_MOCAP.subject()}:Left_Foot")
            left_ankle_norm_hip = data_MOCAP.normalized(f"{data_MOCAP.subject()}:Left_Hip", f"{data_MOCAP.subject()}:Left_Ankle")
            left_knee_norm_hip = data_MOCAP.normalized(f"{data_MOCAP.subject()}:Left_Hip", f"{data_MOCAP.subject()}:Left_Knee")

            
            right_foot_norm_hip=data_MOCAP.normalized(f"{data_MOCAP.subject()}:Right_Hip", f"{data_MOCAP.subject()}:Right_Foot")
            right_ankle_norm_hip = data_MOCAP.normalized(f"{data_MOCAP.subject()}:Right_Hip", f"{data_MOCAP.subject()}:Right_Ankle")
            right_knee_norm_hip = data_MOCAP.normalized(f"{data_MOCAP.subject()}:Right_Hip", f"{data_MOCAP.subject()}:Right_Knee")

                        
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
            
            distance_from_obstacle_x = data_MOCAP.distance_x_from(f"{data_MOCAP.subject()}:Back1", f"{data_MOCAP.subject()}:Obstacle1")
            
            # Distance from start
            distance_from_start =  data_MOCAP.distance_from(f"{data_MOCAP.subject()}:Back1", f"{data_MOCAP.subject()}:Platform1")
        
            # Distance from end
            distance_from_end = data_MOCAP.distance_from(f"{data_MOCAP.subject()}:Back1", f"{data_MOCAP.subject()}:Platform2")
        
            
            #Setup a array with all parameters
            
            data = {
                'left_foot_x' : left_foot[1],
                'left_foot_y' : left_foot[0],
                'left_foot_z' : left_foot[2],
                
                'left_ankle_x' : left_ankle[1],
                'left_ankle_y' : left_ankle[0],
                'left_ankle_z' : left_ankle[2],
                
                'left_knee_x' : left_knee[1],
                'left_knee_y' : left_knee[0],
                'left_knee_z' : left_knee[2],
                
                'left_hip_x' : left_hip[1],
                'left_hip_y' : left_hip[0],
                'left_hip_z' : left_hip[2],
                
                'right_foot_x' : right_foot[1],
                'right_foot_y' : right_foot[0],
                'right_foot_z' : right_foot[2],
                
                'right_ankle_x' : right_ankle[1],
                'right_ankle_y' : right_ankle[0],
                'right_ankle_z' : right_ankle[2],
                
                'right_knee_x' : right_knee[1],
                'right_knee_y' : right_knee[0],
                'right_knee_z' : right_knee[2],
                
                'right_hip_x' : right_hip[1],
                'right_hip_y' : right_hip[0],
                'right_hip_z' : right_hip[2],
                
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
                
                'left_foot_x_norm_hip' : left_foot_norm_hip[1],
                'left_foot_y_norm_hip' : left_foot_norm_hip[0],
                'left_foot_z_norm_hip' : left_foot_norm_hip[2],
                
                'left_ankle_x_norm_hip' : left_ankle_norm_hip[1],
                'left_ankle_y_norm_hip' : left_ankle_norm_hip[0],
                'left_ankle_z_norm_hip' : left_ankle_norm_hip[2],
                
                'left_knee_x_norm_hip' : left_knee_norm_hip[1],
                'left_knee_y_norm_hip' : left_knee_norm_hip[0],
                'left_knee_z_norm_hip' : left_knee_norm_hip[2],
                
                
                'right_foot_x_norm_hip' : right_foot_norm_hip[1],
                'right_foot_y_norm_hip' : right_foot_norm_hip[0],
                'right_foot_z_norm_hip' : right_foot_norm_hip[2],
                
                'right_ankle_x_norm_hip' : right_ankle_norm_hip[1],
                'right_ankle_y_norm_hip' : right_ankle_norm_hip[0],
                'right_ankle_z_norm_hip' : right_ankle_norm_hip[2],
                
                'right_knee_x_norm_hip' : right_knee_norm_hip[1],
                'right_knee_y_norm_hip' : right_knee_norm_hip[0],
                'right_knee_z_norm_hip' : right_knee_norm_hip[2],
                
                
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
                'distance_from_obstacle_x' : distance_from_obstacle_x,
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
            
    
            
        
        
    if Stance == True:
        save_path_stance = rf"{save_folder}/Stances_{idx[0]}_{idx[1]}_{idx[2]}.xlsx"
        
        if os.path.isfile(save_path_stance) and force_rewrite == False:
            print(rf"{save_path_stance} already exists")
        else:
            stance_left = data_MOCAP.stance(f"{data_MOCAP.subject()}:Left_Foot")
            stance_right = data_MOCAP.stance(f"{data_MOCAP.subject()}:Right_Foot")
            
            stance_left = [list(arr) for arr in stance_left]
            stance_right = [list(arr) for arr in stance_right]
            
            # Déterminez la longueur maximale
            max_length = max(max(len(l) for l in stance_right), max(len(l) for l in stance_left))
            
            # Égalisez les longueurs
            for i in range(len(stance_left)):
                stance_left[i].extend([np.nan]*(max_length - len(stance_left[i])))
            
            for i in range(len(stance_right)):
                stance_right[i].extend([np.nan]*(max_length - len(stance_right[i])))



            data_stance = {
                "stance_left_idx": stance_left[0],
                "stance_left_x": stance_left[1],
                "stance_left_y": stance_left[2],
                "stance_left_z": stance_left[3],
                "stance_left_lengths": stance_left[4],
            
                "stance_right_idx": stance_right[0],
                "stance_right_x": stance_right[1],
                "stance_right_y": stance_right[2],
                "stance_right_z": stance_right[3],
                "stance_right_lengths": stance_right[4]
            }
            
            df_stance = pd.DataFrame(data_stance)
            
            df_stance.to_excel(save_path_stance)
                
            
    if State == True:      
        save_path_states = rf"{save_folder}/States_{idx[0]}_{idx[1]}_{idx[2]}.xlsx"
        
        if os.path.isfile(save_path_states) and force_rewrite == False:
            print(rf"{save_path_states} already exists")
        
        else:
                
            
            def detect_swing_states(speed):
                # Convert speed to swing index
                foot_swing = (np.array(speed) > 0.035).astype(int)
                foot_swing_idx = np.where(foot_swing == 1)[0]
                
                # Detect bouts
                bouts = []
                start = foot_swing_idx[0]
                for i in range(1, len(foot_swing_idx)):
                    if foot_swing_idx[i] != foot_swing_idx[i-1] + 1:  # If not contiguous
                        bouts.append((start, foot_swing_idx[i-1]))  # End of the current bout
                        start = foot_swing_idx[i]  # Start of the next bout
                bouts.append((start, foot_swing_idx[-1]))  # Add the last bout
                
                # Remove single entry bouts and bouts less than 5 long
                bouts = [bout for bout in bouts if bout[0] != bout[1] and bout[1] - bout[0] + 1 >= 5]
                
                # Fuse close bouts
                fused_bouts = []
                prev_bout = bouts[0]
                for curr_bout in bouts[1:]:
                    # Check if current bout is close to the previous bout
                    if curr_bout[0] <= prev_bout[1] + 2:
                        # Fuse bouts
                        prev_bout = (prev_bout[0], curr_bout[1])
                    else:
                        # Add the previous bout to the fused bouts list
                        fused_bouts.append(prev_bout)
                        prev_bout = curr_bout
                
                # Add the last bout
                fused_bouts.append(prev_bout)
                
                return fused_bouts
    
    
    
            left_foot_swing = detect_swing_states(data_MOCAP.speed(f"{data_MOCAP.subject()}:Left_Foot"))
            right_foot_swing = detect_swing_states(data_MOCAP.speed(f"{data_MOCAP.subject()}:Right_Foot"))
    
    
            
            if control_plot == True:
                left_foot = data_MOCAP.coord(f"{data_MOCAP.subject()}:Left_Foot")
                right_foot = data_MOCAP.coord(f"{data_MOCAP.subject()}:Right_Foot")
    
                plt.figure()
                plt.plot(-left_foot[1],left_foot[2], color='r')
                for swing in left_foot_swing:
                    plt.axvspan(-left_foot[1][swing[0]],-left_foot[1][swing[1]],alpha=0.3)
                
                plt.figure()
                plt.plot(left_foot[2], color='r')
                for swing in left_foot_swing:
                    plt.axvspan(swing[0],swing[1],alpha=0.3)
                    
                
                plt.figure()
                plt.plot(-right_foot[1],right_foot[2], color='b')
                for swing in right_foot_swing:
                    plt.axvspan(-right_foot[1][swing[0]],-right_foot[1][swing[1]],alpha=0.3)
                
                plt.figure()
                plt.plot(right_foot[2], color='b')
                for swing in right_foot_swing:
                    plt.axvspan(swing[0],swing[1],alpha=0.3)
          
    
    print(f"{j+1}/{len(Files)}") 

#%%
                