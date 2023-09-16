# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 08:43:06 2023

@author: Gilles.DELBECQ
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

session_name = '0026_02_08'

# List of mocap files
mocap_files = ["//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0026_01_12_mocap.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0026_01_14_mocap.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0026_01_15_mocap.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0026_01_16_mocap.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0026_01_17_mocap.xlsx"]

# List of rates files
rates_files = ["//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0026_01_12_rates.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0026_01_14_rates.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0026_01_15_rates.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0026_01_16_rates.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0026_01_17_rates.xlsx"]

# Read and concatenate mocap data
all_mocap_data = pd.concat([pd.read_excel(file) for file in mocap_files], ignore_index=True)

# Read and concatenate rates data
all_rates_data = pd.concat([pd.read_excel(file) for file in rates_files], ignore_index=True)




savefig_path = rf"\\equipe2-nas1\Public\DATA\Gilles\Spikesorting_August_2023\SI_Data\spikesorting_results\{session_name}\kilosort3\curated\processing_data\plots\correlations"
os.makedirs(savefig_path, exist_ok=True)

# Calculate the correlation matrix for mocap parameters
mocap_corr_matrix = all_mocap_data.drop(columns=['time_axis']).corr()

# Calculate the correlation matrix for the units
rates_corr_matrix = all_rates_data.drop(columns=['time_axis']).corr()

# Merge the two datasets on 'time_axis'
merged_data = pd.merge(all_mocap_data, all_rates_data, on='time_axis', how='inner')

# Drop the 'time_axis' column as it's not needed for the correlation calculation
merged_data = merged_data.drop(columns=['time_axis'])

# Calculate the correlation matrix between units and mocap parameters
unit_mocap_corr_matrix = merged_data.corr()

# Extract only the correlation between units and mocap parameters
unit_mocap_corr_submatrix = unit_mocap_corr_matrix.loc[all_rates_data.columns[1:], all_mocap_data.columns[1:]]

#%% Heatmaps

# Visualization

# Plotting the heatmap for mocap correlation matrix
plt.figure(figsize=(20, 10))
sns.heatmap(mocap_corr_matrix, cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix - Mocap Parameters")
plt.tight_layout()
plt.show()

plt.savefig(rf"{savefig_path}/mocap_matrix.png")


# Plotting the heatmap for rates correlation matrix
plt.figure(figsize=(20, 10))
sns.heatmap(rates_corr_matrix, cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix - Units")
plt.tight_layout()
plt.show()

plt.savefig(rf"{savefig_path}/rates_matrix.png")

# Plotting the heatmap for unit-mocap correlation submatrix
plt.figure(figsize=(20, 10))
sns.heatmap(unit_mocap_corr_submatrix, cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix - Units vs Mocap Parameters")
plt.tight_layout()
plt.show()

plt.savefig(rf"{savefig_path}/mocap_rates_matrix.png")

#%% Barplot Average correlation global

# Extract the average correlations for each mocap parameter
average_mocap_correlations = mocap_corr_matrix.mean().sort_values(ascending=False)

# Plotting the average correlations for mocap parameters
plt.figure(figsize=(20, 15))
sns.barplot(y=average_mocap_correlations.index, x=average_mocap_correlations.values, palette="viridis")
plt.title("Average Correlations - Mocap Parameters")
plt.xlabel("Average Correlation")
plt.tight_layout()
plt.show()

# Extract the average correlations for each unit
average_unit_correlations = rates_corr_matrix.mean().sort_values(ascending=False)

# Plotting the average correlations for units
plt.figure(figsize=(20, 10))
sns.barplot(y=average_unit_correlations.index, x=average_unit_correlations.values, palette="viridis")
plt.title("Average Correlations - Units")
plt.xlabel("Average Correlation")
plt.tight_layout()
plt.show()


#%% Zscore

# Z-score standardization of the mocap data
for column in all_mocap_data.columns[1:]:
    mean_val = all_mocap_data[column].mean()
    std_val = all_mocap_data[column].std()
    all_mocap_data[column] = (all_mocap_data[column] - mean_val) / std_val

# Z-score standardization of the firing rates
for column in all_rates_data.columns[1:]:
    mean_val = all_rates_data[column].mean()
    std_val = all_rates_data[column].std()
    all_rates_data[column] = (all_rates_data[column] - mean_val) / std_val

# Calculate correlation matrices
mocap_corr_matrix = all_mocap_data.drop(columns=['time_axis']).corr()
rates_corr_matrix = all_rates_data.drop(columns=['time_axis']).corr()
merged_data = pd.merge(all_mocap_data, all_rates_data, on='time_axis', how='inner').drop(columns=['time_axis'])
unit_mocap_corr_matrix = merged_data.corr().loc[all_rates_data.columns[1:], all_mocap_data.columns[1:]]

# Visualization
plt.figure(figsize=(20, 10))
sns.heatmap(mocap_corr_matrix, cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix - Standardized Mocap Parameters")
plt.tight_layout()
plt.show()

plt.savefig(rf"{savefig_path}/z_score_mocap_matrix.png")


plt.figure(figsize=(20, 10))
sns.heatmap(rates_corr_matrix, cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix - Standardized Units")
plt.tight_layout()
plt.show()

plt.savefig(rf"{savefig_path}/z_score_rates_matrix.png")


plt.figure(figsize=(20, 10))
sns.heatmap(unit_mocap_corr_matrix, cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix - Standardized Units vs Standardized Mocap Parameters")
plt.tight_layout()
plt.show()
plt.savefig(rf"{savefig_path}/z_score_mocap_rates_matrix.png")
