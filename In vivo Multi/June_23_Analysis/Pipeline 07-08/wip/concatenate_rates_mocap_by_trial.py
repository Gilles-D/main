# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 14:31:57 2023

@author: Gil
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


#%% Functions
def list_recording_files(path,type_file):
    """
    List all CSV files containing the specified session in the name
    in the specified directory and its subdirectories.

    Parameters:
        path (str): The directory path to search for CSV files.
        session (str): The session to search for in the file names.

    Returns:
        list: A list of paths to CSV files containing the session in their name.
    """
    import os

    csv_files = []
    for folderpath, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.lower().endswith(".xlsx") and type_file in filename:
                csv_files.append(os.path.join(folderpath, filename))

    return csv_files

#%% Parameters

session = "0022_01_08"

sync_data_path = rf"\\equipe2-nas1\Public\DATA\Gilles\Spikesorting_August_2023\SI_Data\spikesorting_results\{session}\kilosort3\curated\processing_data\sync_data_rate_sigma_20.0 msms_Gaussian"
mocap_session_name = "0022_01"


list_of_trials = [14, 15, 16, 17, 18, 19, 20, 21, 22]
# list_of_trials = [3,4,5,6,7,8,9,10,11,12,13]

# List of columns to focus on
focus_columns = [
    "left_foot_x", "left_foot_y", "left_foot_z", 
    "left_ankle_x", "left_ankle_y", "left_ankle_z", 
    "left_knee_x", "left_knee_y", "left_knee_z",
    "left_hip_x", "left_hip_y", "left_hip_z", 
    "right_foot_x", "right_foot_y", "right_foot_z", 
    "right_ankle_x", "right_ankle_y", "right_ankle_z",
    "right_knee_x", "right_knee_y", "right_knee_z", 
    "right_hip_x", "right_hip_y", "right_hip_z", 
    
    "left_foot_x_norm", "left_foot_y_norm", "left_foot_z_norm",
    "left_ankle_x_norm", "left_ankle_y_norm", "left_ankle_z_norm",
    "left_knee_x_norm", "left_knee_y_norm", "left_knee_z_norm", 
    "left_hip_x_norm", "left_hip_y_norm", "left_hip_z_norm",
    "right_foot_x_norm", "right_foot_y_norm", "right_foot_z_norm",
    "right_ankle_x_norm","right_ankle_y_norm", "right_ankle_z_norm",
    "right_knee_x_norm", "right_knee_y_norm", "right_knee_z_norm",
    "right_hip_x_norm", "right_hip_y_norm", "right_hip_z_norm",
    
    "left_ankle_angle", "left_knee_angle", "left_hip_angle",
    "right_ankle_angle", "right_knee_angle", "right_hip_angle",
    
    "back1_x", "back1_y", "back1_z", "back2_x", "back2_y", "back2_z",
    "back_orientation", "back_inclination",
    
    "speed_back1", "speed_left_foot", "speed_right_foot", 
    "obstacle_x", "obstacle_y", "obstacle_z", 

    "distance_from_obstacle", "distance_from_obstacle_x", 
    # "distance_from_start", "distance_from_end"
]




#%% Script

concatenated_df=None

for trial in list_of_trials:
    print(rf'Trial {trial}')
    rates_df = pd.read_excel(rf"{sync_data_path}/{mocap_session_name}_{trial}_rates.xlsx")
    mocap_df = pd.read_excel(rf"{sync_data_path}/{mocap_session_name}_{trial}_mocap.xlsx")
    
    mocap_df = mocap_df[["time_axis"] + focus_columns]
    
    merged_df = pd.merge(mocap_df, rates_df, on="time_axis")
    
    
    # Filter mocap_df to only include rows from the first non-missing "back_1_Z" value to the last non-missing "back_1_Z" value
    filtered_mocap_df = merged_df[merged_df['back1_z'].first_valid_index(): merged_df['back1_z'].last_valid_index() + 1]
    
    
    #TODO : wtf?
    
    # nan_counts = filtered_mocap_df.isna().sum()
    # top_5_columns_with_nan = nan_counts.nlargest(30).index.tolist()
    # top_5_nan_counts = nan_counts.nlargest(30).values
    
    # for i,col in enumerate(top_5_columns_with_nan):
    #     print(rf"{col} : {top_5_nan_counts[i]/len(filtered_mocap_df)*100} % nans")
    
    # cleaned_focus_df = filtered_mocap_df.dropna()
    
    # print(rf"{round(100 - (len(merged_df) - len(cleaned_focus_df) )/len(merged_df)*100,2)} % of rows left")
    
    
    if concatenated_df is not None:
        concatenated_df = pd.concat([concatenated_df,filtered_mocap_df])
    else:
        concatenated_df = filtered_mocap_df



import tkinter as tk
from tkinter import filedialog

# Créez une fenêtre principale Tkinter (facultatif)
root = tk.Tk()
root.withdraw()  # Masquer la fenêtre principale

# Utilisez le dialogue de sauvegarde de fichier
file_path = filedialog.asksaveasfilename(
    defaultextension=".xlsx",
    filetypes=[("Fichiers Excel", "*.xlsx"), ("Tous les fichiers", "*.*")]
)

      
concatenated_df.to_excel(file_path)

# N'oubliez pas de détruire la fenêtre principale (si vous l'avez créée) à la fin
root.destroy()