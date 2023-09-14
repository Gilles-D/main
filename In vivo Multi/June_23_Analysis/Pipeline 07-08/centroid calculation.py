# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 18:15:57 2023

@author: Gilles.DELBECQ
"""

import pandas as pd

# Load the Excel data
data = pd.read_excel("//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0022_01_3_mocap.xlsx")

# List of marker coordinates
markers = [
    'left_foot_x', 'left_foot_y', 'left_foot_z',
    'left_ankle_x', 'left_ankle_y', 'left_ankle_z',
    'left_knee_x', 'left_knee_y', 'left_knee_z',
    'left_hip_x', 'left_hip_y', 'left_hip_z',
    
    'right_foot_x', 'right_foot_y', 'right_foot_z',
    'right_ankle_x', 'right_ankle_y', 'right_ankle_z',
    'right_knee_x', 'right_knee_y', 'right_knee_z',
    'right_hip_x', 'right_hip_y', 'right_hip_z',
    
    'back1_x', 'back1_y', 'back1_z',
    'back2_x', 'back2_y', 'back2_z'
]

# Calculate centroid for each time point
centroids = data[markers].apply(lambda row: {
    'centroid_x': row[[i for i in markers if "_x" in i]].mean(),
    'centroid_y': row[[i for i in markers if "_y" in i]].mean(),
    'centroid_z': row[[i for i in markers if "_z" in i]].mean()
}, axis=1, result_type='expand')

# Merge centroids with the original data
data_with_centroids = pd.concat([data, centroids], axis=1)

# Display the first few rows of the data with calculated centroids
data_with_centroids[['time_axis', 'centroid_x', 'centroid_y', 'centroid_z']].head()

