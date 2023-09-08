# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 09:46:04 2023

@author: MOCAP
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the provided excel files
optotag_infos_session_1 = pd.read_excel("D:/ePhy/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/optotag_infos.xlsx")
units_metrics_session_1 = pd.read_excel("D:/ePhy/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/units_metrics.xlsx")
optotag_infos_session_2 = pd.read_excel("D:/ePhy/SI_Data/spikesorting_results/0026_05_08/kilosort3/curated/processing_data/optotag_infos.xlsx")
units_metrics_session_2 = pd.read_excel("D:/ePhy/SI_Data/spikesorting_results/0026_05_08/kilosort3/curated/processing_data/units_metrics.xlsx")


def compute_euclidean_distance(df1, df2, columns):
    """Compute pairwise Euclidean distances between rows of df1 and df2 based on specified columns."""
    distances = np.zeros((len(df1), len(df2)))
    for i, row1 in df1.iterrows():
        for j, row2 in df2.iterrows():
            distances[i, j] = np.linalg.norm(row1[columns].values - row2[columns].values)
    return distances

# Compute spatial distances
spatial_columns = ['Unit position x', 'Unit depth']
spatial_distances = compute_euclidean_distance(units_metrics_session_1, units_metrics_session_2, spatial_columns)

# Compute waveform distances
waveform_columns = ['peak_to_valley', 'peak_trough_ratio', 'half_width', 'repolarization_slope', 'recovery_slope']
waveform_distances = compute_euclidean_distance(units_metrics_session_1, units_metrics_session_2, waveform_columns)

similarity_waveform = 1 / (waveform_distances + 1e-6)

similarity_spatial = 1 / (spatial_distances + 1e-6)

# Additional analysis can be added as needed

# Extract unit names (identifiers) for both sessions
unit_names_session_1 = units_metrics_session_1['Unit'].values
unit_names_session_2 = units_metrics_session_2['Unit'].values

# Visualize the similarity based on waveform parameters using unit names
plt.figure(figsize=(15, 10))
sns.heatmap(similarity_waveform, cmap="YlGnBu", cbar_kws={'label': 'Similarity'}, 
            xticklabels=unit_names_session_2, yticklabels=unit_names_session_1)
plt.title("Similarity Degree based on Waveform Parameters")
plt.xlabel("Units from Session 2")
plt.ylabel("Units from Session 1")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# Visualize the similarity based on spatial location using unit names
plt.figure(figsize=(15, 10))
sns.heatmap(similarity_spatial, cmap="YlGnBu", cbar_kws={'label': 'Similarity'}, 
            xticklabels=unit_names_session_2, yticklabels=unit_names_session_1)
plt.title("Similarity Degree based on Spatial Location")
plt.xlabel("Units from Session 2")
plt.ylabel("Units from Session 1")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()


# Merge the optotag_infos data with the units_metrics data based on unit identifiers for both sessions
merged_session_1 = pd.merge(units_metrics_session_1, optotag_infos_session_1, left_on="Unit", right_on="units", how="left")
merged_session_2 = pd.merge(units_metrics_session_2, optotag_infos_session_2, left_on="Unit", right_on="units", how="left")

# Update the waveform columns to include the new parameters
extended_waveform_columns = ['peak_to_valley', 'peak_trough_ratio', 'half_width', 'repolarization_slope', 
                             'recovery_slope', 'reliability_scores', 'delays', 'jitters']

# Compute distances based on the extended waveform parameters
extended_waveform_distances = compute_euclidean_distance(merged_session_1, merged_session_2, extended_waveform_columns)

# Compute the similarity from distances for the extended waveform parameters
similarity_extended_waveform = 1 / (extended_waveform_distances + 1e-6)

# Visualize the updated similarity based on extended waveform parameters using unit names
plt.figure(figsize=(15, 10))
sns.heatmap(similarity_extended_waveform, cmap="YlGnBu", cbar_kws={'label': 'Similarity'}, 
            xticklabels=unit_names_session_2, yticklabels=unit_names_session_1)
plt.title("Similarity Degree based on Extended Waveform Parameters")
plt.xlabel("Units from Session 2")
plt.ylabel("Units from Session 1")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

