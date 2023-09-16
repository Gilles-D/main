# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 17:19:12 2023

@author: Gil
"""

# Complete code with English comments

# 1. Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

#%% functions

def list_recording_files(path):
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
            if filename.lower().endswith(".xlsx") and "mocap" in filename:
                csv_files.append(os.path.join(folderpath, filename))

    return csv_files

def Check_Save_Dir(save_path):
    """
    Check if the save folder exists. If not, create it.

    Args:
        save_path (str): Path to the save folder.

    """
    import os
    isExist = os.path.exists(save_path)
    if not isExist:
        os.makedirs(save_path)  # Create folder for the experiment if it does not already exist

    return

#%% 2. Load the datasets
spike_times = pd.read_excel(r"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/0022_01_08_spike_times.xlsx")

mocap_files = list_recording_files(r'//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian')


save_folder = r"\\equipe2-nas1\Public\DATA\Gilles\Spikesorting_August_2023\SI_Data\spikesorting_results\0022_01_08\kilosort3\curated\processing_data\plots/Speed_Rasterplots" 


for file in mocap_files:
    trial = file.split('_')[-2]
    session = file.split('_')[-3]
    mocap_data = pd.read_excel(file)
    
    # 3. Filter the mocap_data to include all rows between the first and last non-NaN speed_back1
    filtered_mocap_data = mocap_data.dropna(subset=['speed_back1'])
    start_index = filtered_mocap_data.index[0]
    end_index = filtered_mocap_data.index[-1]
    subsampled_mocap_data = mocap_data.iloc[start_index:end_index+1]
    
    # 4. Begin plotting
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 4.1. Plot the speed_back1 data on the top subplot
    axs[0].plot(subsampled_mocap_data['time_axis'], subsampled_mocap_data['speed_back1'], color='blue')
    axs[0].set_title(rf'Speed_back1 {session}_{trial}')
    axs[0].set_ylabel('Speed')
    
    # 4.2. Create a raster plot for spike times on the bottom subplot
    for idx, column in enumerate(spike_times.columns[1:]):
        times = spike_times[column].dropna().values
        # Only include spike times within the range of the mocap data
        times = times[(times >= subsampled_mocap_data['time_axis'].iloc[0]) & (times <= subsampled_mocap_data['time_axis'].iloc[-1])]
        axs[1].scatter(times, [idx] * len(times), marker='|', color='black')
    
    # 4.3. Adjust raster plot aesthetics
    axs[1].set_yticks(range(len(spike_times.columns[1:])))
    axs[1].set_yticklabels(spike_times.columns[1:])
    axs[1].set_title('Raster plot of spike times')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Units')
    
    # 5. Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
    
    save_path = rf"{save_folder}/raster_{session}_{trial}"
    Check_Save_Dir(os.path.dirname(save_path))
    plt.savefig(save_path)
    
    
    
    
    
    # 1. Define parameters for binning
    bin_size = 0.05  # 50ms
    time_bins = np.arange(subsampled_mocap_data['time_axis'].iloc[0], 
                          subsampled_mocap_data['time_axis'].iloc[-1] + bin_size, 
                          bin_size)
    
    # 2. Initialize an empty matrix to hold binned spike counts
    spike_counts = np.zeros((len(spike_times.columns[1:]), len(time_bins) - 1))
    
    # 3. Bin spike times and count them
    for idx, column in enumerate(spike_times.columns[1:]):
        times = spike_times[column].dropna().values
        # Only include spike times within the range of the mocap data
        times = times[(times >= subsampled_mocap_data['time_axis'].iloc[0]) & (times <= subsampled_mocap_data['time_axis'].iloc[-1])]
        counts, _ = np.histogram(times, bins=time_bins)
        spike_counts[idx, :] = counts

    # 4. Begin plotting
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 4.1. Plot the speed_back1 data on the top subplot
    axs[0].plot(subsampled_mocap_data['time_axis'], subsampled_mocap_data['speed_back1'], color='blue')
    axs[0].set_title('Evolution of speed_back1 over time (Updated Data)')
    axs[0].set_ylabel('Speed')
    
    # 4.2. Create a heatmap for binned spike counts on the bottom subplot
    axs[1].imshow(spike_counts, aspect='auto', interpolation='none', origin='lower',
                  extent=[time_bins[0], time_bins[-1], 0, len(spike_times.columns[1:])],
                  cmap='viridis')
    
    # 4.3. Adjust heatmap aesthetics
    axs[1].set_yticks(range(len(spike_times.columns[1:])))
    axs[1].set_yticklabels(spike_times.columns[1:])
    axs[1].set_title('Heatmap of spike counts (50ms bins)')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Units')
    
    # 5. Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
    
    save_path = rf"{save_folder}/heatmap_{session}_{trial}"
    Check_Save_Dir(save_path)
    plt.savefig(save_path)