# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 10:19:58 2023

@author: Gil
"""

#%% Parameters
session = "0022_01_08"
processing_data_path = rf"G:\Data\ePhy\{session}\processing_data"

instantaneous_rate_bin_size = 1 #s
trageted_instantaneous_rate_bin_size = 0.005 #s

plot_check = True
units_to_plot = ["Unit_70",'Unit_7','Unit_2', "Unit_6", "Unit_8", "Unit_20"]


#%% Loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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




# Load the spike times data
spike_times = pd.read_excel(rf"{processing_data_path}/spike_times.xlsx")


#TODO : loop on all mocap trials
# Load the mocap data

mocap_data = pd.read_excel(rf"{processing_data_path}/Mocap_data_catwalk.xlsx")



#%% Compute instaneous rate (on 1s window)
# Creating 1-second time bins
bin_edges = np.arange(mocap_data["time_axis"].min(), mocap_data["time_axis"].max() + 1, instantaneous_rate_bin_size)

# Compute the smoothed firing rate for each unit using 1-second bins
firing_rates = pd.DataFrame()
firing_rates["time_axis"] = (bin_edges[:-1] + bin_edges[1:]) / 2  # Bin centers

for column in spike_times.columns[1:]:
    # Count the number of spikes in each 1-second bin for the given unit
    spike_counts, _ = np.histogram(spike_times[column].dropna(), bins=bin_edges)
    
    # Compute firing rate (spikes/second)
    firing_rate = spike_counts  # Since bin width is 1 second, rate = count
    firing_rates[column] = firing_rate

if plot_check == True:
    # Plotting the smoothed firing rates for the selected units
    plt.figure(figsize=(15, 10))
    for unit in units_to_plot:
        plt.plot(firing_rates["time_axis"], firing_rates[unit], label=unit)
    
    plt.xlabel("Time (s)")
    plt.ylabel("Firing Rate (spikes/s)")
    plt.title("Smoothed Firing Rates of Selected Units Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#%% Interpolate to get 5ms sampling period
# Creating a new time axis with 5ms interval
new_time_axis = np.arange(firing_rates["time_axis"].min(), firing_rates["time_axis"].max(), trageted_instantaneous_rate_bin_size)

# Interpolating the firing rates onto the new time axis
interpolated_rates = pd.DataFrame()
interpolated_rates["time_axis"] = new_time_axis

for column in firing_rates.columns[1:]:
    interpolated_values = np.interp(new_time_axis, firing_rates["time_axis"], firing_rates[column])
    interpolated_rates[column] = interpolated_values

if plot_check == True:
    # Plotting the smoothed firing rates for the selected units
    plt.figure(figsize=(15, 10))
    for unit in units_to_plot:
        plt.plot(interpolated_rates["time_axis"], interpolated_rates[unit], label=unit)
    
    plt.xlabel("Time (s)")
    plt.ylabel("Firing Rate (spikes/s)")
    plt.title("Interpolated Firing Rates of Selected Units Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#%%Rounding time axis to 3 decimals
interpolated_rates['time_axis'] = interpolated_rates['time_axis'].round(3)
mocap_data['time_axis'] = mocap_data['time_axis'].round(3)

common_axis = np.intersect1d(interpolated_rates['time_axis'], mocap_data['time_axis'])

print(rf"common axis reach {round(len(common_axis)/len(mocap_data['time_axis'])*100,2)}% of mocap axis")
