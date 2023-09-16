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



# List of marker coordinates for hips and back
hip_back_markers = [
    'left_hip_x', 'left_hip_y', 'left_hip_z',
    'right_hip_x', 'right_hip_y', 'right_hip_z',
    'back1_x', 'back1_y', 'back1_z',
    'back2_x', 'back2_y', 'back2_z'
]

# Calculate centroid for each time point based on hips and back markers
hip_back_centroids = data[hip_back_markers].apply(lambda row: {
    'hip_back_centroid_x': row[[i for i in hip_back_markers if "_x" in i]].mean(),
    'hip_back_centroid_y': row[[i for i in hip_back_markers if "_y" in i]].mean(),
    'hip_back_centroid_z': row[[i for i in hip_back_markers if "_z" in i]].mean()
}, axis=1, result_type='expand')

# Merge the new centroids with the original data
data_with_hip_back_centroids = pd.concat([data_with_centroids, hip_back_centroids], axis=1)

# Display the first few rows of the data with calculated centroids for hips and back
data_with_hip_back_centroids[['time_axis', 'hip_back_centroid_x', 'hip_back_centroid_y', 'hip_back_centroid_z']].head()




from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Determine the global min and max for the centroids across all dimensions
global_min = min(data_with_centroids[['centroid_x', 'centroid_y', 'centroid_z']].min())
global_max = max(data_with_centroids[['centroid_x', 'centroid_y', 'centroid_z']].max())


# Plotting the centroids, hips-back centroid, and foot trajectories in 3D space with the same scale on all axes

plt.figure(figsize=(14, 12))
ax = plt.axes(projection='3d')

# Plotting the 3D trajectory of the original centroid
ax.plot3D(data_with_hip_back_centroids['centroid_x'], data_with_hip_back_centroids['centroid_y'], data_with_hip_back_centroids['centroid_z'], color='blue', label='Original Centroid')

# Plotting the 3D trajectory of the hips-back centroid
ax.plot3D(data_with_hip_back_centroids['hip_back_centroid_x'], data_with_hip_back_centroids['hip_back_centroid_y'], data_with_hip_back_centroids['hip_back_centroid_z'], color='purple', linestyle='-', label='Hips-Back Centroid')

# Plotting the 3D trajectory of the right foot
ax.plot3D(data_with_hip_back_centroids['right_foot_x'], data_with_hip_back_centroids['right_foot_y'], data_with_hip_back_centroids['right_foot_z'], color='red', linestyle='--', label='Right Foot')

# Plotting the 3D trajectory of the left foot
ax.plot3D(data_with_hip_back_centroids['left_foot_x'], data_with_hip_back_centroids['left_foot_y'], data_with_hip_back_centroids['left_foot_z'], color='green', linestyle='--', label='Left Foot')

# Setting equal scales
ax.set_xlim([global_min, global_max])
ax.set_ylim([global_min, global_max])
ax.set_zlim([global_min, global_max])

# Setting labels, title and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Trajectories of the Centroids and Feet over Time (Equal Axis Scales)')
ax.legend()

plt.show()
