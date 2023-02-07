# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 13:29:01 2022
@author: Gilles.DELBECQ


Uses MOCAP analysis class
Flatten coordinates in Z axis according to the start and end of the platform Zs


to do
    #Détecter pics bas
    #prendre les 2 premiers
    #calculer coef dir
    #flatten

"""

import os
import sys
import pandas as pd
import numpy as np



"""
------------------------------PARAMETERS------------------------------
"""
#Load MOCAP class
sys.path.append(r'D:\Gilles.DELBECQ\GitHub\main\MOCAP\Analysis')
import MOCAP_analysis_class as MA

#Directory location of raw csvs
root_dir=r'//equipe2-nas1/Gilles.DELBECQ/Data/MOCAP/Cohorte 2/CSV_gap_filled'

#Directory location to save flattened csv
save_dir=r'//equipe2-nas1/Gilles.DELBECQ/Data/MOCAP/Cohorte 2/CSV_gap_filled_flat'


# #Location of the data_info file
# data_info_path = '//equipe2-nas1/Gilles.DELBECQ/Data/MOCAP/Cohorte 2/Session_Data.xlsx'
# data_info = MA.DATA_file(data_info_path)





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
    
    #Use class function to flatten the coords for both feet using the 2 reference points on the platform
    left_foot_flat = np.transpose(data_MOCAP.flatten(f"{data_MOCAP.subject()}:Left_Foot"))
    right_foot_flat = np.transpose(data_MOCAP.flatten(f"{data_MOCAP.subject()}:Right_Foot"))
    
    
    #Saves it as a dataframe
    data=np.concatenate((left_foot_flat,right_foot_flat),axis=1)
    df = pd.DataFrame(data)
    names=[f"{data_MOCAP.subject()}:Left_Foot_X",f"{data_MOCAP.subject()}:Left_Foot_Y",f"{data_MOCAP.subject()}:Left_Foot_Z",
           f"{data_MOCAP.subject()}:Right_Foot_X",f"{data_MOCAP.subject()}:Right_Foot_Y",f"{data_MOCAP.subject()}:Right_Foot_Z"]
    df.columns = names
    
    #Saves it in a csv
    MA.Check_Save_Dir(save_dir)
    df.to_csv(rf"{save_dir}\{data_MOCAP.subject()}_{data_MOCAP.session_idx()}_{data_MOCAP.trial_idx()}.csv") 
    
    print(f"{i}/{len(Files)}")
    i=i+1