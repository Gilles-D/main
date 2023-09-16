# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 11:11:43 2023

@author: Gilles.DELBECQ
"""

"""
Two diffrent plotting : one with Analysis excel file and the other with the mocap data time axised

"""




import pandas as pd

# Load the data from the provided Excel files
analysis_df = pd.read_excel('//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/mocap_files/Auto-comp/0022/Analysis/Analysis_0022_01_07.xlsx')
stances_df = pd.read_excel('//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/mocap_files/Auto-comp/0022/Analysis/Stances_0022_01_07.xlsx')

analysis_df.head(), stances_df.head()

import matplotlib.pyplot as plt

#%% Z en fonction de X

# Extract relevant data for the left foot
left_foot_x = analysis_df['left_foot_x']
left_foot_z = analysis_df['left_foot_z']
stance_left_x = stances_df['stance_left_x']

# Plot the positions Z in function of X for the left foot
plt.figure(figsize=(12, 6))
plt.plot(left_foot_x, left_foot_z, label='Left Foot Z Position', color='blue')

# Add vertical lines for the stance positions
for x in stance_left_x:
    plt.axvline(x=x, color='green', linestyle='--', alpha=0.7)

plt.title("Positions Z en fonction de X pour le pied gauche")
plt.xlabel("Position X")
plt.ylabel("Position Z")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Extract relevant data for the right foot
right_foot_x = analysis_df['right_foot_x']
right_foot_z = analysis_df['right_foot_z']
stance_right_x = stances_df['stance_right_x']

# Plot the positions Z in function of X for the right foot
plt.figure(figsize=(12, 6))
plt.plot(right_foot_x, right_foot_z, label='Right Foot Z Position', color='red')

# Add vertical lines for the stance positions
for x in stance_right_x:
    plt.axvline(x=x, color='green', linestyle='--', alpha=0.7)

plt.title("Positions Z en fonction de X pour le pied droit")
plt.xlabel("Position X")
plt.ylabel("Position Z")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% X en fonction du temps

# Load the stances file
stances_data = pd.read_excel("//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/mocap_stances/0022_01_9_stances.xlsx")

# Load the mocap file
mocap_data = pd.read_excel("//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0022_01_9_mocap.xlsx")

# Plot right_foot_x_norm in function of time
plt.figure(figsize=(15, 6))
plt.plot(mocap_data['time_axis'], mocap_data['right_foot_z_norm'], label='Right Foot X Coordinate', color='blue')

# Add vertical lines for right_foot_stance times
for stance_time in stances_data['stance_right_times']:
    plt.axvline(x=stance_time, color='red', linestyle='--')

plt.title('Right Foot X Coordinate over Time with Stance Times')
plt.xlabel('Time')
plt.ylabel('Right Foot X Coordinate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

