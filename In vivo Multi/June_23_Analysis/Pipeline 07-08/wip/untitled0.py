# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 14:42:52 2023

@author: MOCAP
"""

import pandas as pd

def calculate_depths(file_paths):
    # List to hold data from each file
    data_frames = []
    
    # Load each file and append to list
    for file_path in file_paths:
        df = pd.read_excel(file_path)
        data_frames.append(df)
    
    # Concatenate all dataframes into a single one
    concatenated_data = pd.concat(data_frames, ignore_index=True)
    
    # Filter for 'OPTO' true and false
    opto_true = concatenated_data[concatenated_data['OPTO'] == True]
    opto_false = concatenated_data[concatenated_data['OPTO'] == False]
    
    # Calculate mean and std for 'OPTO' true
    mean_depth_true = opto_true['Unit depth'].mean()
    std_depth_true = opto_true['Unit depth'].std()
    
    # Calculate mean and std for 'OPTO' false
    mean_depth_false = opto_false['Unit depth'].mean()
    std_depth_false = opto_false['Unit depth'].std()
    
    return {
        'opto_true': {'mean_depth': mean_depth_true, 'std_depth': std_depth_true},
        'opto_false': {'mean_depth': mean_depth_false, 'std_depth': std_depth_false}
    }

# For now, we'll use the same file for demonstration purposes
file_paths = [
    'D:/ePhy/SI_Data/spikesorting_results/0026_29_07/kilosort3/curated/processing_data/units_data.xlsx',
    'D:/ePhy/SI_Data/spikesorting_results/0026_01_08/kilosort3/curated/processing_data/units_data.xlsx',
    'D:/ePhy/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/units_data.xlsx',
    'D:/ePhy/SI_Data/spikesorting_results/0023_28_07/kilosort3/curated/processing_data/units_data.xlsx',
    'D:/ePhy/SI_Data/spikesorting_results/0023_31_07/kilosort3/curated/processing_data/units_data.xlsx',
    'D:/ePhy/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/units_data.xlsx',
    'D:/ePhy/SI_Data/spikesorting_results/0022_28_07/kilosort3/curated/processing_data/units_data.xlsx',
    'D:/ePhy/SI_Data/spikesorting_results/0022_31_07/kilosort3/curated/processing_data/units_data.xlsx',
    'D:/ePhy/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/units_data.xlsx'
   
    ]  # Replace with the actual list of file paths
depth_statistics = calculate_depths(file_paths)
depth_statistics
