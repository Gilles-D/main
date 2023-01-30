# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 11:15:08 2022
@author: Gilles.DELBECQ


Uses MOCAP analysis class
Plot left and right feet of raw csv
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
sys.path.append(r'D:\Gilles.DELBECQ\GitHub\main\MOCAP\Analysis')
import MOCAP_analysis_class as MA


#Directories location
root_dir='//equipe2-nas1/Gilles.DELBECQ/Data/MOCAP/Cohorte 2/Raw_CSV_no_completion'
flat_csv_path='//equipe2-nas1/Gilles.DELBECQ/Data/MOCAP/Cohorte 2/Raw_CSV_flat'


# #Location of the data_info file
# data_info_path = '//equipe2-nas1/Gilles.DELBECQ/Data/MOCAP/Cohorte 2/Session_Data.xlsx'
# data_info = MA.DATA_file(data_info_path)


#Location of the figures to save
savefig_path=r'\\equipe2-nas1\Gilles.DELBECQ\Data\MOCAP\Cohorte 2\Figs\Raw_traj'

Save = True
Save_extension = "png" #svg or png

Flat = True






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
i=1








"""
------------------------------MAIN LOOP------------------------------
"""

for file in Files:
    #Load csv file
    data_MOCAP = MA.MOCAP_file(file)
    
    #Get Trial, session idexes
    idx = data_MOCAP.whole_idx()
    
    #Get coords for each foot
    left_foot=data_MOCAP.coord(f"{data_MOCAP.subject()}:Left_Foot")
    right_foot=data_MOCAP.coord(f"{data_MOCAP.subject()}:Right_Foot")
    

    #Plot trajectory of each foot
    figure1 = plt.figure()
    plt.title(f'Raw Feet trajectory {data_MOCAP.subject()}_{data_MOCAP.session_idx()}_{data_MOCAP.trial_idx()}')
    plt.plot(-left_foot[1],left_foot[2],color='red',label='Left')
    plt.plot(-right_foot[1],right_foot[2],color='blue',label='Right')
    #Plot a vertical line for the IR Beam
    plt.axvline(-stat.median(data_MOCAP.coord(f"{data_MOCAP.subject()}:IR Beam1")[1]))
    
    plt.ylim(0,50)
    
    if Save == True:
        MA.Check_Save_Dir(savefig_path)
        plt.savefig(f"{savefig_path}/{idx[0]}_{idx[1]}_{idx[2]}.{Save_extension}")
        plt.close('all')
    
    if Flat == True:
        flat_coords=pd.read_csv(f"{flat_csv_path}/{idx[0]}_{idx[1]}_{idx[2]}.csv")
        figure2=plt.figure()
        plt.title(f'Raw Feet trajectory flatten {data_MOCAP.subject()}_{data_MOCAP.session_idx()}_{data_MOCAP.trial_idx()}')
        plt.plot(flat_coords[f"{idx[0]}:Left_Foot_X"].tolist(),flat_coords[f"{idx[0]}:Left_Foot_Z"].tolist(),color='red',label='Left')
        plt.plot(flat_coords[f"{idx[0]}:Right_Foot_X"].tolist(),flat_coords[f"{idx[0]}:Right_Foot_Z"].tolist(),color='blue',label='Right')
        
        plt.gca().invert_xaxis()
        plt.axvline(stat.median(data_MOCAP.coord(f"{data_MOCAP.subject()}:IR Beam1")[1]))
        
        plt.ylim(0,50)
        if Save == True:
            MA.Check_Save_Dir(rf"{savefig_path}/Flat")
            plt.savefig(f"{savefig_path}/Flat/{idx[0]}_{idx[1]}_{idx[2]}_flat.{Save_extension}")
            plt.close('all')
    
    print(f"{i}/{len(Files)}")
    i=i+1
