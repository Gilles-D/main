# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 17:28:12 2023

@author: Gilles.DELBECQ
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load spike times data
spike_times = pd.read_excel(r"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/spike_times.xlsx")
spike_units = spike_times.columns[1:]

# Fictitious mocap file paths (to be replaced with real paths)
mocap_paths = ["//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0022_01_14_mocap.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0022_01_15_mocap.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0022_01_16_mocap.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0022_01_17_mocap.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0022_01_18_mocap.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0022_01_19_mocap.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0022_01_20_mocap.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0022_01_21_mocap.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0022_01_22_mocap.xlsx"]

def interpolate_values_final(spike_times, mocap_data):
    interpolated_values = []
    
    # Filter out mocap data where distance_from_obstacle_x is NaN
    mocap_data_filtered = mocap_data.dropna(subset=['distance_from_obstacle_x'])
    first_valid_time = mocap_data_filtered['time_axis'].iloc[0]
    last_valid_time = mocap_data_filtered['time_axis'].iloc[-1]
    
    # Filter spike times to only consider those within the valid mocap time range
    spike_times = spike_times[(spike_times >= first_valid_time) & (spike_times <= last_valid_time)]
    
    for spike in spike_times:
        before = mocap_data_filtered[mocap_data_filtered['time_axis'] <= spike].iloc[-1]
        after = mocap_data_filtered[mocap_data_filtered['time_axis'] >= spike].iloc[0]
        
        if before['time_axis'] == spike:
            interpolated_values.append(before['distance_from_obstacle_x'])
        elif before['time_axis'] < spike < after['time_axis']:
            fraction = (spike - before['time_axis']) / (after['time_axis'] - before['time_axis'])
            distance = before['distance_from_obstacle_x'] + fraction * (after['distance_from_obstacle_x'] - before['distance_from_obstacle_x'])
            interpolated_values.append(distance)
        else:
            interpolated_values.append(np.nan)
    
    return interpolated_values

def plot_histogram_for_unit(unit_data, unit_name):
    plt.figure(figsize=(10,6))
    plt.hist(unit_data, bins=60, edgecolor='k', alpha=0.7)
    plt.title(f"Histogram of Interpolated distance_from_obstacle_x for {unit_name}")
    plt.xlabel("Interpolated distance_from_obstacle_x")
    plt.ylabel("Number of Spikes")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.gca().invert_xaxis()  # Inversion de l'axe des x


# Initialize a dictionary to hold cumulative interpolated values for each unit
cumulative_interpolated_values = {unit: [] for unit in spike_units}

# Loop through each mocap file path
for mocap_path in mocap_paths:
    try:
        current_mocap_data = pd.read_excel(mocap_path)
        for unit in spike_units:
            unit_data = spike_times[unit].dropna()
            interpolated_values = interpolate_values_final(unit_data, current_mocap_data)
            cumulative_interpolated_values[unit].extend(interpolated_values)
    except:
        print(f"Error loading or processing {mocap_path}. Skipping to the next.")

# Create histograms for all units using the cumulative values
for unit, values in cumulative_interpolated_values.items():
    plot_histogram_for_unit(values, unit)

plt.show()
