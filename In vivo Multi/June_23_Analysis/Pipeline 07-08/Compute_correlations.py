# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 14:20:52 2023

@author: Gilles.DELBECQ
"""

#%% Parameters
session = "0022_01_08"
processing_data_path = rf"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/{session}/kilosort3/curated/processing_data"

plots_extension = "png"

units_to_plot = ["Unit_70",'Unit_7','Unit_2', "Unit_6", "Unit_8", "Unit_20"]

plot_chek = True
do_save_dataframes = True

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

# Load the mocap data
mocap_data = pd.read_excel(rf"{processing_data_path}/Mocap_data_catwalk.xlsx")



#%% Compute instaneous rate (on 1s window)
# Creating 1-second time bins
bin_edges_1s = np.arange(mocap_data["time_axis"].min(), mocap_data["time_axis"].max() + 1, 1)

# Compute the smoothed firing rate for each unit using 1-second bins
firing_rates_1s = pd.DataFrame()
firing_rates_1s["time_axis"] = (bin_edges_1s[:-1] + bin_edges_1s[1:]) / 2  # Bin centers

for column in spike_times.columns[1:]:
    # Count the number of spikes in each 1-second bin for the given unit
    spike_counts, _ = np.histogram(spike_times[column].dropna(), bins=bin_edges_1s)
    
    # Compute firing rate (spikes/second)
    firing_rate = spike_counts  # Since bin width is 1 second, rate = count
    firing_rates_1s[column] = firing_rate

if plot_chek == True:
    # Plotting the smoothed firing rates for the selected units
    plt.figure(figsize=(15, 10))
    for unit in units_to_plot:
        plt.plot(firing_rates_1s["time_axis"], firing_rates_1s[unit], label=unit)
    
    plt.xlabel("Time (s)")
    plt.ylabel("Firing Rate (spikes/s)")
    plt.title("Smoothed Firing Rates of Selected Units Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#%% Interpolate to get 5ms sampling period
# Creating a new time axis with 5ms interval
new_time_axis = np.arange(firing_rates_1s["time_axis"].min(), firing_rates_1s["time_axis"].max(), 0.005)

# Interpolating the firing rates onto the new time axis
interpolated_rates = pd.DataFrame()
interpolated_rates["time_axis"] = new_time_axis

for column in firing_rates_1s.columns[1:]:
    interpolated_values = np.interp(new_time_axis, firing_rates_1s["time_axis"], firing_rates_1s[column])
    interpolated_rates[column] = interpolated_values

if plot_chek == True:
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



#%%Perform correlation
# Merging the interpolated firing rates with the mocap_data based on the time axis
synchronized_data = pd.merge_asof(interpolated_rates, mocap_data, on="time_axis", direction="nearest")

# Computing correlations between firing rates and motion capture parameters
correlation_matrix = synchronized_data.corr()

# Extracting correlations between units and motion capture parameters
units = interpolated_rates.columns[1:]
mocap_parameters = mocap_data.columns[1:]

correlations = correlation_matrix.loc[units, mocap_parameters]
df_correlations = pd.DataFrame(correlations)

if do_save_dataframes == True:
    df_correlations.to_excel(rf"{processing_data_path}/correlations.xlsx")


#%% All parameters heatmap
import seaborn as sns

# Plotting a heatmap of the correlations
plt.figure(figsize=(20, 15))
sns.heatmap(correlations, cmap="coolwarm", center=0, annot=False, fmt=".2f", cbar_kws={'label': 'Pearson Correlation Coefficient'})
plt.title("Correlation between Units' Firing Rates and Motion Capture Parameters")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


savepath = rf"{processing_data_path}\plots\correlation_heatmap_whole.{plots_extension}"
Check_Save_Dir(os.path.dirname((savepath)))
plt.savefig(savepath)

#%%Separated heatmaps
# Segregating the motion capture parameters based on the categories specified

# Parameters related to _x, _y, _z positions and _angle
position_angle_params = [col for col in mocap_parameters if "_x" in col or "_y" in col or "_z" in col or "_angle" in col]
correlation_position_angle = correlations[position_angle_params]

# Parameters related to speed and distance
speed_distance_params = [col for col in mocap_parameters if "speed" in col or "distance" in col]
correlation_speed_distance = correlations[speed_distance_params]

# Parameters related to back
back_params = [col for col in mocap_parameters if "back" in col]
correlation_back = correlations[back_params]

# Plotting the heatmaps

# Heatmap for _x, _y, _z positions and _angle parameters
plt.figure(figsize=(20, 15))
sns.heatmap(correlation_position_angle, cmap="coolwarm", center=0, annot=True, fmt=".2f", cbar_kws={'label': 'Pearson Correlation Coefficient'})
plt.title("Correlation between Units' Firing Rates and Position & Angle Parameters")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

savepath = rf"{processing_data_path}\plots\correlation_heatmap_pos_angles.{plots_extension}"
Check_Save_Dir(os.path.dirname((savepath)))
plt.savefig(savepath)

# Heatmap for speed and distance parameters
plt.figure(figsize=(20, 15))
sns.heatmap(correlation_speed_distance, cmap="coolwarm", center=0, annot=True, fmt=".2f", cbar_kws={'label': 'Pearson Correlation Coefficient'})
plt.title("Correlation between Units' Firing Rates and Speed & Distance Parameters")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

savepath = rf"{processing_data_path}\plots\correlation_heatmap_speed_distance.{plots_extension}"
Check_Save_Dir(os.path.dirname((savepath)))
plt.savefig(savepath)

# Heatmap for back parameters
plt.figure(figsize=(20, 15))
sns.heatmap(correlation_back, cmap="coolwarm", center=0, annot=True, fmt=".2f", cbar_kws={'label': 'Pearson Correlation Coefficient'})
plt.title("Correlation between Units' Firing Rates and Back Parameters")

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

savepath = rf"{processing_data_path}\plots\correlation_heatmap_back.{plots_extension}"
Check_Save_Dir(os.path.dirname((savepath)))
plt.savefig(savepath)

#%% Linear regression for best couple unit/parameter
# Identify the unit and motion capture parameter with the highest absolute correlation
max_corr_value = correlations.abs().max().max()
unit, mocap_param = np.where(correlations.abs() == max_corr_value)

selected_unit = correlations.index[unit[0]]
selected_mocap_param = correlations.columns[mocap_param[0]]

selected_unit, selected_mocap_param


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Removing rows with NaN values for the selected unit and motion capture parameter
cleaned_data = synchronized_data[[selected_unit, selected_mocap_param]].dropna()

# Data preparation
X_cleaned = cleaned_data[selected_mocap_param].values.reshape(-1, 1)
y_cleaned = cleaned_data[selected_unit].values

# Fitting the linear regression model
model_cleaned = LinearRegression().fit(X_cleaned, y_cleaned)

# Predicting values using the model
y_pred_cleaned = model_cleaned.predict(X_cleaned)

# Calculating R^2 score for the cleaned data
r2_cleaned = r2_score(y_cleaned, y_pred_cleaned)

# Plotting the regression line and data points for the cleaned data
plt.figure(figsize=(12, 8))
plt.scatter(X_cleaned, y_cleaned, color='blue', s=10, label="Data Points")
plt.plot(X_cleaned, y_pred_cleaned, color='red', linewidth=2, label=f"Regression Line (R^2 = {r2_cleaned:.2f})")
plt.xlabel(selected_mocap_param)
plt.ylabel(f"Firing Rate of {selected_unit}")
plt.title(f"Regression Analysis (after cleaning NaNs) between {selected_unit} and {selected_mocap_param}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Linear regression with speed
# Selecting a speed-related parameter based on the highest absolute correlation with Unit_6
speed_related_params = [col for col in correlations.columns if "speed" in col]
selected_speed_param = correlations.loc[selected_unit, speed_related_params].abs().idxmax()

# Data preparation for the selected speed parameter
X_speed = synchronized_data[selected_speed_param].dropna().values.reshape(-1, 1)
y_speed = synchronized_data.loc[synchronized_data[selected_speed_param].notna(), selected_unit].values

# Fitting the linear regression model for the selected speed parameter
model_speed = LinearRegression().fit(X_speed, y_speed)

# Predicting values using the model
y_pred_speed = model_speed.predict(X_speed)

# Calculating R^2 score for the selected speed parameter
r2_speed = r2_score(y_speed, y_pred_speed)

# Plotting the regression line and data points for the selected speed parameter
plt.figure(figsize=(12, 8))
plt.scatter(X_speed, y_speed, color='blue', s=10, label="Data Points")
plt.plot(X_speed, y_pred_speed, color='red', linewidth=2, label=f"Regression Line (R^2 = {r2_speed:.2f})")
plt.xlabel(selected_speed_param)
plt.ylabel(f"Firing Rate of {selected_unit}")
plt.title(f"Regression Analysis between {selected_unit} and {selected_speed_param}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Plot all parameters R² for each units

# Adjusting the regression analysis to handle cases with no valid data points
r2_values = {}

for unit in units:
    r2_values[unit] = []
    for param in mocap_parameters:
        X_temp = synchronized_data[param].dropna().values.reshape(-1, 1)
        y_temp = synchronized_data.loc[synchronized_data[param].notna(), unit].values
        
        # Only proceed with regression if there are valid data points
        if len(X_temp) > 0 and len(y_temp) > 0:
            # Fitting the linear regression model
            model_temp = LinearRegression().fit(X_temp, y_temp)
            
            # Predicting values using the model
            y_pred_temp = model_temp.predict(X_temp)
            
            # Extracting R^2 value
            r2_temp = r2_score(y_temp, y_pred_temp)
        else:
            r2_temp = np.nan  # Set to NaN if no valid data points
        
        r2_values[unit].append(r2_temp)

# Converting the results to a DataFrame
r2_df = pd.DataFrame(r2_values, index=mocap_parameters)

# Plotting the R^2 values for each unit in separate horizontal bar plots
for unit in units:
    plt.figure(figsize=(10, 15))
    r2_df[unit].sort_values().plot(kind='barh', color='skyblue')
    plt.xlabel('R^2 Value')
    plt.ylabel('Motion Capture Parameters')
    plt.title(f'R^2 Values for {unit}')
    plt.tight_layout()
    plt.show()
    
    savepath = rf"{processing_data_path}\plots\regressions\regression_{unit}.{plots_extension}"
    Check_Save_Dir(os.path.dirname((savepath)))
    plt.savefig(savepath)
    
#%% Time lagged cross-correlation
from scipy.signal import correlate

# Extracting data for Unit_6 and speed_y
unit_data = synchronized_data[selected_unit].dropna()
speed_data = synchronized_data.loc[unit_data.index, selected_speed_param]

# Normalizing the data to have zero mean and unit variance
unit_data_normalized = (unit_data - unit_data.mean()) / unit_data.std()
speed_data_normalized = (speed_data - speed_data.mean()) / speed_data.std()

# Compute the cross-correlation
lags = np.arange(-len(unit_data) + 1, len(unit_data))
cross_corr = correlate(unit_data_normalized, speed_data_normalized, mode='full')

# Plotting the cross-correlation
plt.figure(figsize=(15, 7))
plt.plot(lags * 0.05, cross_corr, label="Cross-Correlation")  # Multiply lags by 0.05 to convert to seconds
plt.axvline(0, color='red', linestyle='--', label="Zero Lag")
plt.title(f"Time-lagged Cross-Correlation between {selected_unit} and {selected_speed_param}")
plt.xlabel("Time Lag (s)")
plt.ylabel("Cross-Correlation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#%% Clustering : PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Preparing the data for PCA: Using the firing rates of all units
data_for_pca = synchronized_data[units].dropna()

# Standardizing the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_for_pca)

# Applying PCA
pca = PCA()
pca_data = pca.fit_transform(scaled_data)

# Plotting the explained variance by each principal component
explained_variance = pca.explained_variance_ratio_.cumsum()

plt.figure(figsize=(12, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance by Principal Components")
plt.grid(True)
plt.tight_layout()
plt.show()

#%% kmeans determine clusters
from sklearn.cluster import KMeans

# Using the first few principal components that capture significant variance
pca_data_reduced = pca_data[:, :5]

# Determining the optimal number of clusters using the elbow method
inertia = []
for k in range(1, 15):  # We'll test up to 14 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pca_data_reduced)
    inertia.append(kmeans.inertia_)

# Plotting the elbow curve
plt.figure(figsize=(12, 6))
plt.plot(range(1, 15), inertia, marker='o', linestyle='--')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal Number of Clusters")
plt.grid(True)
plt.tight_layout()
plt.show()

#%% kmeans clustering
# Applying K-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(pca_data_reduced)

# Visualizing the clusters in the first two principal components
plt.figure(figsize=(12, 8))
for i in range(3):
    plt.scatter(pca_data_reduced[cluster_labels == i, 0], pca_data_reduced[cluster_labels == i, 1], label=f"Cluster {i+1}")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label="Cluster Centers")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Clusters in PCA-reduced Space")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
from sklearn.cluster import KMeans

# Extracting the firing rates of units from the synchronized_data
cluster_data = synchronized_data[units].dropna()

# Using the elbow method to find the optimal number of clusters
wcss = []  # Within-cluster sum of squares
n_clusters_range = range(1, 11)  # Checking for 1 to 10 clusters

for i in n_clusters_range:
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(cluster_data)
    wcss.append(kmeans.inertia_)

# Plotting the elbow method graph
plt.figure(figsize=(10, 6))
plt.plot(n_clusters_range, wcss, marker='o')
plt.title('Elbow Method For Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()
