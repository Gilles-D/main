# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 10:58:38 2023

Plot the stacked trajectories for each animal
Normalized trajectories from the start platform marker


@author: Gilles.DELBECQ
"""

import os
import sys
import pandas as pd
from matplotlib import pyplot as plt 
import statistics as stat


"""
------------------------------PARAMETERS------------------------------
"""
#Load MOCAP class
sys.path.append(r'C:\Users\Gil\Documents\GitHub\main\MOCAP\Analysis')
import MOCAP_analysis_class as MA


#Directories location
root_dir=r'D:\Seafile\Seafile\Ma bibliothèque\Data\MOCAP\Cohorte2\CSV_gap_filled'
flat_csv_path=r'D:\Seafile\Seafile\Ma bibliothèque\Data\MOCAP\Cohorte2\CSV_gap_filled_flat'

#Sessions idexes to analyse (as a list)
sessions_to_analyze = [1,3]




#Location of the data_info file
data_info_path = 'C:/Users/Gil/Documents/GitHub/main/MOCAP/Data/Session_Data_Cohorte2.xlsx'
data_info = MA.DATA_file(data_info_path)


#Location of the figures to save
savefig_path=r'D:\Seafile\Seafile\Ma bibliothèque\Data\MOCAP\Cohorte2\Figs/Stacked/'

Save = False
Save_extension = "png" #svg or png


"""
-----------------------------FUCTIONS-----------------------------------

"""
def flatten(marker,start,stop):
    """
    Parameters
    ----------
    marker : TYPE
        DESCRIPTION.

    Returns
    -------
    new_coords : TYPE
        np.array([y,x,new_z]).

    """
    
    import statistics as stat
    import numpy as np
    
    # start = coord(f"{self.subject()}:Platform1")
    # stop = coord(f"{self.subject()}:Platform2")
    
    start_x,start_z=stat.median(start[1]),stat.median(start[2])
    stop_x,stop_z=stat.median(stop[1]),stat.median(stop[2]) 
    
    coef_dir = (stop_z-start_z)/(stop_x-start_x)
    
    # print(coef_dir)
    # if coef_dir >= 0:
    #     print('+')
    # else:
    #     print('-')
    
    x = marker[1]
    y = marker[0]
    
    new_z = []
    
    for i in range(len(x)):
        if coef_dir >= 0:
            shift = x[i]*coef_dir
            new_z.append(marker[2][i]-shift)
            
        else:
            shift = marker[1][i]*coef_dir
            new_z.append(marker[2][i]-shift)
            
    
    new_z = np.array(new_z)
    
    new_coords = np.array([x,y,new_z])
    
    return new_coords




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

def moving_average(data, window_size=5):
    result=[]
    
    moving_sum = sum(data[:window_size])
    result.append(moving_sum/window_size)
    
    for i in range(len(data)-window_size):
        moving_sum += (data[i+window_size]-data[i])
        result.append(moving_sum/window_size)
    return result


for animal in animals:
    print(animal)
    figure1=plt.figure()
    plt.title(rf"{animal} stacked traces")    
    for file in Files:
        data_MOCAP = MA.MOCAP_file(file)
        
        if data_MOCAP.subject() == animal:
            if int(data_MOCAP.session_idx()) in sessions_to_analyze: #Takes only first 2 sessions (basic locomotion)
                print(file)
                
                #Takes file info
                idx = data_MOCAP.whole_idx()
                info = data_info.get_info(data_MOCAP.subject(),data_MOCAP.session_idx(),
                                          data_MOCAP.trial_idx())
                
                
                speed = data_MOCAP.speed(f"{data_MOCAP.subject()}:Back1")
                speed_smooth=moving_average(speed)
                
                plt.plot(speed, color="blue")
                
                # if int(data_MOCAP.session_idx()) == 1:
                #     plt.plot(speed_smooth, color="blue")
                # else:
                #     plt.plot(speed_smooth, color="red")

                
                """
                #Get beam coord
                beam = stat.median(data_MOCAP.coord(f"{data_MOCAP.subject()}:IR Beam1")[1])
                
                
                #Normalizer les coordonnées !
                normalized_left_foot=data_MOCAP.normalized(f"{data_MOCAP.subject()}:IR Beam1", f"{idx[0]}:Left_Foot")
                normalized_right_foot=data_MOCAP.normalized(f"{data_MOCAP.subject()}:IR Beam1", f"{idx[0]}:Right_Foot")

                normalized_left_foot_flatten=flatten(normalized_left_foot,data_MOCAP.coord(f"{data_MOCAP.subject()}:Platform1"),data_MOCAP.coord(f"{data_MOCAP.subject()}:Platform2"))
                normalized_right_foot_flatten=flatten(normalized_right_foot,data_MOCAP.coord(f"{data_MOCAP.subject()}:Platform1"),data_MOCAP.coord(f"{data_MOCAP.subject()}:Platform2"))
                
                plt.plot(normalized_left_foot_flatten[0],normalized_left_foot_flatten[2],color='red')
                plt.plot(normalized_right_foot_flatten[0],normalized_right_foot_flatten[2],color='blue')
"""

    if Save == True:
        MA.Check_Save_Dir(savefig_path)
        plt.savefig(f"{savefig_path}/{animal}_stacked (session {sessions_to_analyze[0]} to {sessions_to_analyze[-1]}).{Save_extension}")
        plt.close('all')

