# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 14:31:57 2023

@author: Gil
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


# Load the two excel files again
rates_df = pd.read_excel("G:/Data/ePhy/0022_01_08/processing_data/sync_data/0022_01_3_rates.xlsx")
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
random_units = np.random.choice(rates_df.columns[2:], 5, replace=False)

selected_units = ['Unit_7', 'Unit_74', 'Unit_20', 'Unit_42', 'Unit_62']


# Dictionary to store GLM results for each unit
glm_results = {}

# Loop through each random unit and perform GLM
for unit in selected_units:
    # Merge the unit data with the focus data
    data = pd.merge(cleaned_focus_df, rates_df[["time_axis", unit]], on="time_axis")
    
    # Define dependent and independent variables
    X = data[focus_columns]
    X = sm.add_constant(X)  # Add a constant (intercept) to the predictors
    y = data[unit]
    
    # Fit GLM
    model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    
    # Store the model summary in the results dictionary
    glm_results[unit] = model.summary()

