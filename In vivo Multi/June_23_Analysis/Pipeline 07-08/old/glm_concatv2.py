# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 14:23:59 2023

@author: Gilles.DELBECQ
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# Load the dataset
data = pd.read_excel("//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/concatenated_mocap_rates_catwalk.xlsx")

# Check for missing values
missing_values = data.isnull().sum()

# Display columns with missing values, if any
missing_values[missing_values > 0]

# Splitting the dataset into motion capture features and firing rates

# Columns for firing rates
unit_cols_new = [col for col in data.columns if col.startswith("Unit_")]
unit_data_new = data[unit_cols_new]

# Columns for motion capture features (excluding 'Unnamed: 0' and 'time_axis')
motion_capture_cols_new = [col for col in data.columns if col not in unit_cols_new and col not in ["Unnamed: 0", "time_axis"]]
motion_capture_data_new = data[motion_capture_cols_new]



# Normalize the new motion capture features
scaler = StandardScaler()
motion_capture_normalized_new = scaler.fit_transform(motion_capture_data_new)

# Convert normalized data back to a DataFrame for better readability
motion_capture_normalized_new_df = pd.DataFrame(motion_capture_normalized_new, columns=motion_capture_data_new.columns)

# Fit a GLM for the first unit using the new motion capture data
X_new = sm.add_constant(motion_capture_normalized_new_df)
y_new = unit_data_new['Unit_0']

glm_model_new = sm.GLM(y_new, X_new, family=sm.families.Poisson()).fit()

# Display the summary for the model
glm_model_new.summary()


#%%Visualization 

import matplotlib.pyplot as plt
import numpy as np

# Extract coefficients and p-values
coefficients = glm_model_new.params
p_values = glm_model_new.pvalues

# Sort by absolute value of coefficients for better visualization
sorted_idx = np.abs(coefficients).sort_values(ascending=False).index

# Plot Coefficient values
plt.figure(figsize=(12, 10))
coefficients[sorted_idx].plot(kind='barh')
plt.title('Coefficient Plot for Unit_0')
plt.xlabel('Coefficient Value')
plt.ylabel('Predictors')
plt.show()

# Plot p-values
plt.figure(figsize=(12, 10))
p_values[sorted_idx].plot(kind='barh', color='skyblue')
plt.axvline(x=0.05, color='red', linestyle='--')  # significance level at 0.05
plt.title('P-value Plot for Unit_0')
plt.xlabel('P-value')
plt.ylabel('Predictors')
plt.show()

#%%Check for multicolinearity : VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for each predictor
vif_data = pd.DataFrame()
vif_data["Predictor"] = motion_capture_normalized_new_df.columns
vif_data["VIF"] = [variance_inflation_factor(motion_capture_normalized_new_df.values, i) 
                   for i in range(motion_capture_normalized_new_df.shape[1])]

# Sort by VIF value for better visualization
vif_data_sorted = vif_data.sort_values(by="VIF", ascending=False)

vif_data_sorted


body_parts = set([col.split('_')[0] for col in motion_capture_cols_new if '_' in col])

