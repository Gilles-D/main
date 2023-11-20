# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 14:42:52 2023

@author: MOCAP
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def read_data(file_paths):
    data_frames=[]
    # Load each file and append to list
    for file_path in file_paths:
        df = pd.read_excel(file_path)
        data_frames.append(df)
    
    # Concatenate all dataframes into a single one
    concatenated_data = pd.concat(data_frames, ignore_index=True)
    
    
    return concatenated_data


def calculate_depths_and_plot(file_paths, adjust=0):
    concatenated_data = read_data(file_paths)
    
    # Filter for 'OPTO' true and false
    opto_true = concatenated_data[concatenated_data['OPTO'] == True]
    opto_false = concatenated_data[concatenated_data['OPTO'] == False]
    
    # Subtract 'adjust' from 'Unit depth' for both 'OPTO' true and false
    opto_true['Unit depth'] -= adjust
    opto_false['Unit depth'] -= adjust

    # Update the concatenated data
    concatenated_data = pd.concat([opto_true, opto_false])

    # Calculate mean and std for 'OPTO' true
    mean_depth_true = opto_true['Unit depth'].mean()
    std_depth_true = opto_true['Unit depth'].std()
    
    # Calculate mean and std for 'OPTO' false
    mean_depth_false = opto_false['Unit depth'].mean()
    std_depth_false = opto_false['Unit depth'].std()
    
    # Create the violin plots
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="OPTO", y="corrected_depth", data=concatenated_data, split=True, inner="quart", linewidth=1.5)
    plt.ylim(-1200,-200)
    plt.title('Distribution of Neuron Depths for Opto True vs False (Adjusted)')
    plt.show()
    
    return {
        'opto_true': {'mean_depth': mean_depth_true, 'std_depth': std_depth_true},
        'opto_false': {'mean_depth': mean_depth_false, 'std_depth': std_depth_false}
    }

# For now, we'll use the same file for demonstration purposes
file_paths = [
    # 'D:/ePhy/SI_Data/spikesorting_results/0026_29_07/kilosort3/curated/processing_data/units_data.xlsx',
    # 'D:/ePhy/SI_Data/spikesorting_results/0026_01_08/kilosort3/curated/processing_data/units_data.xlsx',
    # 'D:/ePhy/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/units_data.xlsx',
    # 'D:/ePhy/SI_Data/spikesorting_results/0023_28_07/kilosort3/curated/processing_data/units_data.xlsx',
    # 'D:/ePhy/SI_Data/spikesorting_results/0023_31_07/kilosort3/curated/processing_data/units_data.xlsx',
    # 'D:/ePhy/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/units_data.xlsx',
    # 'D:/ePhy/SI_Data/spikesorting_results/0022_28_07/kilosort3/curated/processing_data/units_data.xlsx',
    # 'D:/ePhy/SI_Data/spikesorting_results/0022_31_07/kilosort3/curated/processing_data/units_data.xlsx',
    # 'D:/ePhy/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/units_data.xlsx'
# "D:/ePhy/SI_Data/spikesorting_results/unit_data_0026.xlsx",
"D:/ePhy/SI_Data/spikesorting_results/unit_data_0022.xlsx"
   
    ]  # Replace with the actual list of file paths
depth_statistics = calculate_depths_and_plot(file_paths, adjust=-145)
depth_statistics




data = read_data(file_paths)
# Scatter plot of all units "peak_to_valley" versus "half_width"
plt.figure(figsize=(10, 6))
sns.scatterplot(x='half_width', y='peak_to_valley',hue='OPTO', data=data)

plt.title('Scatter Plot of Peak to Valley vs Half Width')
plt.xlabel('Half Width (ms)')
plt.ylabel('Peak to Valley (mV)')

# Show the plot
plt.show()

#%% PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# Extract the relevant features
features = ['peak_to_valley', 'peak_trough_ratio', 'half_width', 'repolarization_slope', 'recovery_slope']
X = data[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=2)  # reduce to 2 dimensions for visualization purposes
X_pca = pca.fit_transform(X_scaled)

# Now let's perform the hierarchical clustering using the PCA-transformed data
Z = linkage(X_pca, 'ward')

# Prepare for plotting
plt.figure(figsize=(10, 8))

# Plotting the dendrogram
dendrogram(Z)

plt.title('Hierarchical Clustering Dendrogram (Truncated)')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')

# Show the plot
plt.show()

# Plot the PCA-transformed features as a scatter plot, colored by 'OPTO' tagging
plt.figure(figsize=(10, 6))

# Extract the boolean 'OPTO' column to use as labels for coloring
opto_labels = data['OPTO'].map({True: 'Opto True', False: 'Opto False'}).values

# Create a scatter plot with the two principal components by 'OPTO' tagging
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=opto_labels, palette='viridis')

plt.title('PCA of Neuron Features (Colored by Opto Tagging)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Opto Tagging')

# Show the plot
plt.show()
