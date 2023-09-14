# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 11:11:43 2023

@author: Gilles.DELBECQ
"""

import pandas as pd

# Load the data from the provided Excel files
analysis_df = pd.read_excel('//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/mocap_files/Auto-comp/0022/Analysis/Analysis_0022_01_07.xlsx')
stances_df = pd.read_excel('//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/mocap_files/Auto-comp/0022/Analysis/Stances_0022_01_07.xlsx')

analysis_df.head(), stances_df.head()

import matplotlib.pyplot as plt

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
