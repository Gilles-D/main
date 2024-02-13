import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# List of file paths
file_paths = ['D:/Seafile/Seafile/Ma bibliothèque/Rédaction/Manuscrits/Figures/Résultats/Data fig/MOCAP/0022_01_08/phaseresponse/units_data.xlsx',
              'D:/Seafile/Seafile/Ma bibliothèque/Rédaction/Manuscrits/Figures/Résultats/Data fig/MOCAP/0023_01_08/phaseresponse/units_data.xlsx',
              'D:/Seafile/Seafile/Ma bibliothèque/Rédaction/Manuscrits/Figures/Résultats/Data fig/MOCAP/0026_02_08/phaseresponse/units_data.xlsx']  # Remplacez par vos chemins de fichiers
# Initialize an empty list to store the data
all_data = []

# Process each file
for file_path in file_paths:
    # Load the data from the Excel file
    data = pd.read_excel(file_path)

    # Reshape the data so that each unit is a row with two columns: angle and amplitude
    angle_data = data.iloc[0].values
    amplitude_data = data.iloc[1].values
    units = data.columns

    # Create a DataFrame with the reshaped data
    polar_data = pd.DataFrame({
        'angle': angle_data,
        'amplitude': amplitude_data
    }, index=units).reset_index(drop=True)

    # Convert degrees to radians for plotting
    polar_data['angle_rad'] = np.deg2rad(polar_data['angle'])

    # Add the reshaped data to the list
    all_data.append(polar_data)

# Concatenate all data into a single DataFrame
all_polar_data = pd.concat(all_data).reset_index(drop=True)

# Determine the number of clusters using the elbow method
wcss = []
for n in range(1, 11):
    kmeans = KMeans(n_clusters=n, random_state=0).fit(all_polar_data[['amplitude', 'angle_rad']])
    wcss.append(kmeans.inertia_)

# Assume the elbow is at 3 clusters (or determine it programmatically)
optimal_clusters = 3

# Apply k-means to the concatenated dataset with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
all_polar_data['cluster'] = kmeans.fit_predict(all_polar_data[['amplitude', 'angle_rad']])

# Plotting the polar scatter with clusters
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

# Scatter plot for each cluster
for i in range(optimal_clusters):
    cluster_data = all_polar_data[all_polar_data['cluster'] == i]
    ax.scatter(cluster_data['angle_rad'], cluster_data['amplitude'], label=f'Cluster {i+1}', alpha=0.7)

# Set the polar axes
ax.set_theta_zero_location('N')  # Set 0 degrees to the top
ax.set_theta_direction(-1)  # Set the direction of increase to clockwise

# Add grid, title, and labels
ax.grid(True)
plt.title('Polar Scatter Plot with Clusters for All Files')
ax.set_xlabel('Angle [Radians]')
ax.set_ylabel('Amplitude')

# Add legend
ax.legend()

# Show the plot
plt.show()


#%%

# Initialize an empty list to store the data
all_data = []

# Process each file and keep track of the file origin
for file_index, file_path in enumerate(file_paths):
    # Load the data from the Excel file
    data = pd.read_excel(file_path)

    # Reshape the data so that each unit is a row with two columns: angle and amplitude
    angle_data = data.iloc[0].values
    amplitude_data = data.iloc[1].values
    units = data.columns

    # Create a DataFrame with the reshaped data
    polar_data = pd.DataFrame({
        'angle': angle_data,
        'amplitude': amplitude_data,
        'file_origin': [file_index] * len(units)  # Add a column indicating the file origin
    }, index=units).reset_index(drop=True)

    # Convert degrees to radians for plotting
    polar_data['angle_rad'] = np.deg2rad(polar_data['angle'])

    # Add the reshaped data to the list
    all_data.append(polar_data)

# Concatenate all data into a single DataFrame
all_polar_data = pd.concat(all_data).reset_index(drop=True)

# Plotting the polar scatter with different colors for each file origin
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)


# Unique colors for each file origin
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Extend this list if you have more files

# Scatter plot, colored by file origin
for file_index in range(len(file_paths)):
    file_data = all_polar_data[all_polar_data['file_origin'] == file_index]
    ax.scatter(file_data['angle_rad'], file_data['amplitude'], 
               label=f'File {file_index+1}', color=colors[file_index % len(colors)], alpha=0.7)

# Set the polar axes
ax.set_theta_zero_location('N')  # Set 0 degrees to the top
ax.set_theta_direction(-1)  # Set the direction of increase to clockwise

# Add grid, title, and labels
ax.grid(True)
plt.title('Polar Scatter Plot Colored by File Origin')
# ax.set_xlabel('Angle [Radians]')
# ax.set_ylabel('Amplitude')

# Add legend
ax.legend()

# Set the polar axes
ax.set_theta_zero_location('E')  # Set 0 degrees to the right (East)
ax.set_theta_direction(1)  # Keep the direction of increase clockwise

# Show the plot
plt.show()

plt.savefig(rf"D:\Seafile\Seafile\Ma bibliothèque\Rédaction\Manuscrits\Figures\Résultats\Data fig\MOCAP/polarplot_all_units.svg")

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

all_data = []

for file_index, file_path in enumerate(file_paths):
    data = pd.read_excel(file_path)

    angle_data = data.iloc[0].values
    amplitude_data = data.iloc[1].values
    truth_values = data.iloc[2].values  # Assume the truth values are in the third row
    units = data.columns

    polar_data = pd.DataFrame({
        'angle': angle_data,
        'amplitude': amplitude_data,
        'truth_value': truth_values,  # Add a column for the truth value
        'file_origin': [file_index] * len(units)
    }, index=units).reset_index(drop=True)

    polar_data['angle_rad'] = np.deg2rad(polar_data['angle'])

    all_data.append(polar_data)

all_polar_data = pd.concat(all_data).reset_index(drop=True)

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
truth_colors = {True: 'b', False: 'r'}  # Define colors for true and false

for file_index in range(len(file_paths)):
    file_data = all_polar_data[all_polar_data['file_origin'] == file_index]
    
    for truth_value in [True, False]:
        truth_data = file_data[file_data['truth_value'] == truth_value]
        ax.scatter(truth_data['angle_rad'], truth_data['amplitude'],
                   label=f'File {file_index+1} - {"True" if truth_value else "False"}',
                   color=truth_colors[truth_value], alpha=0.7)

ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)

ax.grid(True)
plt.title('Polar Scatter Plot Colored by File Origin and Truth Value')
ax.set_xlabel('Angle [Radians]')
ax.set_ylabel('Amplitude')
ax.legend()

plt.show()
