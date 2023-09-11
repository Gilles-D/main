# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 15:06:04 2023

@author: Gil
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


# Load the two excel files again
spike_times_df = pd.read_excel("G:/Data/ePhy/0022_01_08/processing_data/spike_times.xlsx")
mocap_df = pd.read_excel("G:/Data/ePhy/0022_01_08/processing_data/sync_data/0022_01_3_mocap.xlsx")


# Filter mocap_df to only include rows from the first non-missing "back_1_Z" value to the last non-missing "back_1_Z" value
filtered_mocap_df = mocap_df[mocap_df['back_1_Z'].first_valid_index(): mocap_df['back_1_Z'].last_valid_index() + 1]

# Interpolate missing values
interpolated_df = filtered_mocap_df.interpolate(method='linear', limit_direction='both')

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
    "left_ankle_angle", "left_knee_angle", "left_hip_angle",
    "right_ankle_angle", "right_knee_angle", "right_hip_angle",
    "back_orientation", "back_inclination", "back_1_Z", "back_2_Z",
    "speed_back1", "speed_left_foot", "speed_right_foot"
]

# Subset the dataframe to only include the focus columns
focus_df = interpolated_df[["time_axis"] + focus_columns]

# Drop rows with missing values in the subset dataframe
cleaned_focus_df = focus_df.dropna()


# Randomly select 5 units from rates_df columns (excluding "time_axis" and "Unnamed: 0" columns)
random_units = np.random.choice(spike_times_df.columns[2:], 5, replace=False)

selected_units = ['Unit_7', 'Unit_74', 'Unit_20', 'Unit_42', 'Unit_62']




def generate_spike_train(spike_times, time_bins):
    """Generate a spike train for given spike times and time bins."""
    # Create a series of zeros for the entire duration
    spike_train = np.zeros_like(time_bins)
    
    # For each spike time, find the closest time bin and set it to 1
    for spike in spike_times:
        # Find the index of the closest time bin to the spike time
        idx = np.argmin(np.abs(time_bins - spike))
        spike_train[idx] = 1
        
    return spike_train

def generate_spike_count(spike_times, time_bins, bin_size):
    """Generate a spike count for given spike times and time bins."""
    spike_counts = np.zeros_like(time_bins)
    
    # For each spike time, find the corresponding time bin and increment the count
    for spike in spike_times:
        idx = np.searchsorted(time_bins, spike)
        if idx < len(time_bins) and abs(spike - time_bins[idx]) <= bin_size / 2:
            spike_counts[idx] += 1
            
    return spike_counts

# Time bins from the filtered motion capture data
time_bins = cleaned_focus_df["time_axis"].values

# Generate spike train for the first random unit (for demonstration)
sample_unit = selected_units[0]
spike_train_sample = generate_spike_train(spike_times_df[sample_unit].dropna().values, time_bins)

bin_size = time_bins[1] - time_bins[0]
spike_count_sample = generate_spike_count(spike_times_df[sample_unit].dropna().values, time_bins, bin_size)




# Dictionary to store GLM results for spike counts
glm_results_spike_counts = {}

# Loop through each random unit and perform GLM using spike counts
for unit in selected_units:
    # Generate spike counts for the unit
    spike_counts = generate_spike_count(spike_times_df[unit].dropna().values, time_bins, bin_size)
    
    # Merge the spike counts with the focus data
    data = cleaned_focus_df.copy()
    data[unit] = spike_counts
    
    # Define dependent and independent variables
    X = data[focus_columns]
    X = sm.add_constant(X)  # Add a constant (intercept) to the predictors
    y = data[unit]
    
    # Fit GLM
    model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    
    # Store the model summary in the results dictionary
    glm_results_spike_counts[unit] = model.summary()

# Displaying the summary for the first unit as an example
glm_results_spike_counts[random_units[0]]
