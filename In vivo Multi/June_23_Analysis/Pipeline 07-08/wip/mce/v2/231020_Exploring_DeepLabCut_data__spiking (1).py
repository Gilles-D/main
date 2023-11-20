# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 09:56:57 2023

@author: Matilde.CORDERO
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy import stats
import random

import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

from matplotlib.colors import LinearSegmentedColormap

# Sampling for spiking
sampling_rate = 20000#Hz

# Sampling period for video in milliseconds
sampling_period_ms = 48


## Dans dossier Spikesorting results
session_results = '0022_31_07'
spikesorting_results_path = r'D:/ePhy/SI_Data/spikesorting_results'
sorter_name = "kilosort3"
sorter_folder = rf'{spikesorting_results_path}/{session_results}/{sorter_name}/curated/processing_data'
spike_times_xls = rf'{sorter_folder}/spike_times.xlsx'
file_instR = "/instantaneous_rates_20ms.xlsx"

## Dans dossier Concatenated (pour temps de stim)
session_name = session_results
concatenated_signals_path = r'D:/ePhy/SI_Data/concatenated_signals'
signal_folder = rf'{concatenated_signals_path}/{session_name}'

## Fichier deep lab cut
DLC = pd.read_excel(r'D:/ePhy/SI_Data/DLC/0022_31_07_2DLC.xlsx', decimal=',')
length = len(DLC)

#refaire un index temps utilisable pour les calculs
index = [i * sampling_period_ms / 1000 for i in range(length)]

DLC = DLC.drop(DLC.columns[0], axis=1)
DLC.index = index
DLC.index.name = 'time' 

#liste des colonnes
print(DLC.columns)

# Create a directory to store the saved figures
save_dir = rf'{sorter_folder}/saved_figures'
os.makedirs(save_dir, exist_ok=True)

# Create a behavior-color dictionary with specific color assignments
behavior_colors = {
    'nan': 'white',
    'nan_behavior' : 'black',
    'Snout rotate right': 'cyan', 
    'Snout rotate left': 'lime', 
    'imobile': 'lightgrey', 
    'stand': 'violet', 
    ' rotate left': 'green', 
    ' rotate right': 'blue', 
    'straight': 'dimgrey', 
    'stand against wall': 'pink',
    'grooming' : 'gold'}


#%%Functions
def Get_recordings_info(session_name, concatenated_signals_path, spikesorting_results_path):
    """
    Cette fonction rÃ©cupÃ¨re les informations d'enregistrement Ã  partir d'un fichier de mÃ©tadonnÃ©es
    dans le dossier de signaux concatÃ©nÃ©s.

    Args:
        session_name (str): Le nom de la session d'enregistrement.
        concatenated_signals_path (str): Le chemin vers le dossier contenant les signaux concatÃ©nÃ©s.
        spikesorting_results_path (str): Le chemin vers le dossier des rÃ©sultats du tri des spikes.

    Returns:
        dict or None: Un dictionnaire contenant les mÃ©tadonnÃ©es si la lecture est rÃ©ussie,
        ou None si la lecture Ã©choue.

    Raises:
        Exception: Si une erreur se produit pendant la lecture du fichier.

    """
    try:
        # Construire le chemin complet vers le fichier de mÃ©tadonnÃ©es
        path = rf'{concatenated_signals_path}/{session_name}/'
        
        # Lire le fichier de mÃ©tadonnÃ©es Ã  l'aide de la bibliothÃ¨que pickle
        print("Lecture du fichier ttl_idx dans le dossier Intan...")
        metadata = pickle.load(open(rf"{path}/ttl_idx.pickle", "rb"))
        
    except Exception as e:
        # GÃ©rer toute exception qui pourrait se produire pendant la lecture du fichier
        print("Aucune information d'enregistrement trouvÃ©e dans le dossier Intan. Veuillez exÃ©cuter l'Ã©tape 0.")
        metadata = None  # Aucune mÃ©tadonnÃ©e disponible en cas d'erreur
    
    print('TerminÃ©')
    return metadata



#%%Loading and aligning

#Load units
recordings_info = Get_recordings_info(session_name,concatenated_signals_path,sorter_folder)

#Load mocap ttl times
# stim_idx = recordings_info['stim_ttl_on'][0::2]
mocap_idx = recordings_info['mocap_ttl_on'][0::2]

##Load instantaneous frequencies
frequfilename = sorter_folder + file_instR
instR_data = pd.read_excel(frequfilename, decimal=',')
instR_data.set_index(instR_data.columns[0], inplace = True)
instR_data.index.name = 'time'  # Set the index name for instR

# instR_data.index = instR_data.index*100

instR = instR_data
instR.index = instR.index/10

#### Alignement DLC sur temps instR ####

tps_LED_on = 267*0.048
tps_TTL = mocap_idx[1] / sampling_rate

DLC.index = DLC.index + (tps_TTL - tps_LED_on)

#### Slicing de instR sur la session d'openField ####

DLC_T1 = min(DLC.index)
DLC_T2 = max(DLC.index)

#Get rid of instR outside OpenField
mask = (instR.index >= DLC_T1) & (instR.index <= DLC_T2)
instR = instR[mask]

# #Time window to be outside of stimulation protocol
# T1 = min(stim_idx) / sampling_rate
# T2 = (max(stim_idx) / sampling_rate) + 5

print(rf"Loading spikesorting results for session {session_name}")
sorter_results = pd.read_excel(spike_times_xls)
sorter_results = sorter_results.drop(sorter_results.columns[0], axis=1)

#Get rid of spike times outside OpenField
# mask = (sorter_results.index >= DLC_T1) & (sorter_results.index <= DLC_T2)
# instR = instR[mask]

# # we = si.load_waveforms(rf'{sorter_folder}/curated/waveforms')

# #Get rid of spikes during stimulation protocol
# for column in sorter_results.columns:
#     # Create a mask for values outside the time range
#     mask = (sorter_results[column] >= T1) & (sorter_results[column] <= T2)
#     # Update the column with NaN for values inside the range
#     sorter_results.loc[mask, column] = np.nan
# sorter_results = sorter_results.dropna(axis=0, how='all')
# #Get rid of instR during stimulation protocol
# mask = (instR.index >= T1) & (instR.index <= T2)
# instR = instR[~mask]

unit_list = sorter_results.columns
# unit_list = sorter_results.columns.str.slice(5,)
print(unit_list)




#%% Extracting from depp lab cut

#calcul vitesse x_tail
DLC['speed_x-Tail_Base'] = DLC['x-Tail_base'].diff().abs()
# filtered_data_tail_x = DLC[DLC['speed_x-Tail_Base'] <= 6]
# filtered_data_tail_x2 = DLC[DLC['llh_Tail_base'] >= 0.6]

DLC['speed_y-Tail_Base'] = DLC['y_Tail_base'].diff().abs()
# filtered_data_tail_y = DLC[DLC['speed_x-Tail_Base'] <= 6]
# filtered_data_tail_y2 = DLC[DLC['llh_Tail_base'] >= 0.6]

DLC['IMM_ARR'] = (DLC['speed_x-Tail_Base'] <= 6) & (DLC['llh_Tail_base'] >= 0.6) & (DLC['speed_y-Tail_Base'] <= 6) & (DLC['llh_Tail_base'] >= 0.6)

DLC['speed_x_Snout'] = DLC['x_Snout'].diff().abs()
# filtered_data_Snout_x = DLC[DLC['speed_x_Snout'] <= 6]
# filtered_data_Snout_x2 = DLC[DLC['llh_Snout'] >= 0.6]

DLC['speed_y_Snout'] = DLC['y_Snout'].diff().abs()
# filtered_data_Snout_y = DLC[DLC['speed_y_Snout'] <= 6]
# filtered_data_Snout_y2 = DLC[DLC['llh_Snout'] >= 0.6]

DLC['IMM_AV'] = (DLC['speed_x_Snout'] <= 6) & (DLC['llh_Snout'] >= 0.6) & (DLC['speed_y_Snout'] <= 6) & (DLC['llh_Snout'] >= 0.6)

DLC['IMMOBILE'] = (DLC['speed_x-Tail_Base'] <= 6) & (DLC['llh_Tail_base'] >= 0.6) & (DLC['speed_y-Tail_Base'] <= 6) & (DLC['llh_Tail_base'] >= 0.6) & (DLC['speed_x_Snout'] <= 6) & (DLC['llh_Snout'] >= 0.6) & (DLC['speed_y_Snout'] <= 6) & (DLC['llh_Snout'] >= 0.6)

DLC['angle'] = DLC.apply(lambda row: np.arctan2(row['y_Snout'] - row["y_Tail_base"], row['x_Snout'] - row["x-Tail_base"]), axis=1)
DLC['angle_change'] = DLC['angle'].diff()


##########################################################################


# Set the threshold for immobility episodes (3 * 48 ms)
threshold = 3 * 0.048

# Initialize lists to store episode details
start_times = []
end_times = []
episode_lengths = []

# Initialize variables to track the current episode
current_episode_start = None
current_episode_end = None

# Iterate through the DataFrame
for index, row in DLC.iterrows():
    if row['IMMOBILE']:
        if current_episode_start is None:
            current_episode_start = index
        current_episode_end = index
    else:
        if current_episode_start is not None:
            episode_length = current_episode_end - current_episode_start
            if episode_length >= threshold:
                start_times.append(current_episode_start)
                end_times.append(current_episode_end)
                episode_lengths.append(episode_length)
            current_episode_start = None
            current_episode_end = None

# Create a new DataFrame for immobility episodes
episode_data = {
    'Start Time': start_times,
    'End Time': end_times,
    'Episode Length': episode_lengths
}

episode_df = pd.DataFrame(episode_data)

# Display the DataFrame of immobility episodes
print(episode_df)



#%% Combining DLC IMMOBILE with inst Frequ

# Create a new column in instR to indicate immobility periods
instR['IMMOBILE'] = False  # Initialize the column to False

# Iterate through the rows of episode_df
for _, row in episode_df.iterrows():
    start_time = row['Start Time']
    end_time = row['End Time']
    
    # Mark the corresponding indices in instR as True for immobility
    instR.loc[(instR.index >= start_time) & (instR.index <= end_time), 'IMMOBILE'] = True


print(instR.groupby('IMMOBILE').mean())

# Loop through the columns (units) in instR
for column in instR.columns:
    # Create a new figure and axis for each unit
    plt.figure()
    ax = sns.kdeplot(instR[instR['IMMOBILE'] == True][column], label='IMMOBILE True')
    ax = sns.kdeplot(instR[instR['IMMOBILE'] == False][column], label='IMMOBILE False')
    ax.set_title(f'Distribution of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Density')
    ax.set_xlim(-5, 20)  # Set the x-axis limits
    plt.legend()
    plt.show()
    

# Define your significance level (alpha)
alpha = 0.05

# Initialize a list to store the results
results = []
p_values = []

mean_imm=[]
mean_mob=[]

# Loop through the columns (neurons) in instR
for column in instR.columns[:-1]:
    # Extract data for the two groups
    immobile_true_data = instR[instR['IMMOBILE'] == True][column]
    immobile_false_data = instR[instR['IMMOBILE'] == False][column]

    # Perform a statistical test (choose between t-test and Mann-Whitney U test)
    t_stat, p_value = stats.ttest_ind(immobile_true_data, immobile_false_data)

    # Check if the p-value is less than alpha (significant)
    if p_value < alpha:
        result = 'SIGN'
    else:
        result = 'NOT SIGN'
        
    p_values.append(p_value)
    results.append(result)
    
    
    mean_imm.append(np.mean(immobile_true_data))
    mean_mob.append(np.mean(immobile_false_data))

"""
Saving in dataframes
"""
df_stats_IMM = pd.DataFrame({
    'mean_imm' : mean_imm,
    'mean_mob' : mean_mob,
    'stats' : results,
    'p' : p_values
    },    index = unit_list)

df_stats_IMM.to_excel(rf"{sorter_folder}/{session_results}_immobile_ttest.xlsx")

#%% Combining inst Frequ with "visual"


# Define the tolerance or range for merging (in milliseconds)
tolerance_ms = 0.048  # Adjust as needed

# Merge only the 'time_index' and 'visual' columns from DLC to instR
instR = pd.merge_asof(instR, DLC, on='time', direction='backward', tolerance=tolerance_ms)
# instR = instR.iloc[:4730,:]
instR = instR.set_index('time')
print(instR.groupby('visual').mean())

# Select only the relevant columns
relevant_columns = instR.filter(like='Unit_').columns.tolist() + ['visual']

# Create a new DataFrame with the relevant columns
heatmap_data = instR[relevant_columns]

# Calculate the mean of each 'Unit_' column
column_means = heatmap_data.iloc[:, :-1].mean()
column_max = heatmap_data.iloc[:, :-1].max()
# Normalize the data by dividing by the mean of each 'Unit_' column
normalized_heatmap_data = 100*heatmap_data.iloc[:, :-1].div(column_means)
norm_max_heatmap_data = 100*heatmap_data.iloc[:, :-1].div(column_max)
# Recombine the 'visual' column with the normalized data
normalized_heatmap_data['visual'] = instR['visual']
norm_max_heatmap_data['visual'] = instR['visual']

pivoted_heatmap_data = heatmap_data.groupby('visual').mean().T
pivoted_normalized_heatmap_data = normalized_heatmap_data.groupby('visual').mean().T
pivoted_norm_max_heatmap_data = norm_max_heatmap_data.groupby('visual').mean().T

# Create the heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(pivoted_normalized_heatmap_data, cmap='bwr', vmin = -500, annot=True, fmt=".2f", linewidths=0.5)
plt.xlabel("Neuron")
plt.ylabel("Visual Behavior")
plt.title("Firing Frequency Heatmap by Visual Behavior")
plt.show()

# plt.figure(figsize=(10, 10))
# sns.heatmap(pivoted_norm_max_heatmap_data, annot=True, fmt=".2f", linewidths=0.5)
# plt.xlabel("Neuron")
# plt.ylabel("Visual Behavior")
# plt.title("Firing Frequency Heatmap by Visual Behavior")
# plt.show()

# Count occurrences of each "visual" value
visual_counts = instR['visual'].value_counts(normalize=True, dropna=False) * 100

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(visual_counts, labels=visual_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Behaviors', y=1.03)

# Display the chart
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


#%% Combining Spike times with "visual" / Raster plot

Unit = 'Unit_37'
spike_times = sorter_results[Unit]

# Select the time window by slicing the sorter_results dataframe
spike_times_window = spike_times[(spike_times >= DLC_T1) & (spike_times <= DLC_T2)]

# Extract "visual" values and time index from DLC
visual_values = DLC['visual']
time_index = DLC.index  # Make sure the index is in seconds or milliseconds



                   

# Define the duration of each segment (60 seconds)
segment_duration = 20
segment_height = 0.2

# Calculate the number of segments
num_segments = int(DLC_T2 - DLC_T1) // segment_duration
spikes_in_segment = []  
rectangles_per_segment = []

# Create subplots for each segment
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Iterate over each segment and create a raster plot
for i in range(num_segments) :
    start_time = DLC_T1 + i * segment_duration
    end_time = DLC_T1 + (i + 1) * segment_duration

    # Select events within the time window
    segment_events = spike_times_window[(spike_times_window >= start_time) & (spike_times_window <= end_time)]

    segment_events_relatif = segment_events - start_time       
    spikes_in_segment.append(segment_events_relatif) 
    
    # Create rectangles for this segment
    segment_rectangles = []

    # Overlay colored squares for each behavior within the segment
    for behavior, color in behavior_colors.items():
        behavior_times = time_index[visual_values == behavior]
        for time in behavior_times:
            if start_time <= time <= end_time:
                time_relative = time - start_time
                # Include color information
                segment_rectangles.append((time_relative, time_relative + 0.048, color))

    rectangles_per_segment.append(segment_rectangles)
    
# Create a figure and axis
fig, ax = plt.subplots()

# Define the height of the rectangles
rect_height = 1
# Initialize a variable to keep track of the vertical offset
vertical_offset = 0

# Iterate over segments and their rectangles
for segment_rectangles in rectangles_per_segment:
    for left, right, color in segment_rectangles:
        ax.fill_betweenx([vertical_offset - rect_height/2 , rect_height/2 + vertical_offset], left, right, color=color)
    vertical_offset += rect_height  # Adjust the offset as needed


# Set axis limits and labels
ax.set_xlim(0, segment_duration)
ax.set_ylim(-rect_height/2, vertical_offset)
ax.set_xlabel('X-Axis')
ax.set_ylabel('Y-Axis')

ax.eventplot(spikes_in_segment, orientation='horizontal', color='black', linelengths=1)
plt.show()



#%% Combining Spike times with "visual" / Histogram

# Extract "visual" values and time index from DLC
visual_values = DLC['visual']
time_index = DLC.index
                  
# Define the duration of each segment (60 seconds)
segment_duration = 20
segment_height = 0.2

# Calculate the number of segments
num_segments = int(DLC_T2 - DLC_T1) // segment_duration


# Create an array to store event frequency within time bins
num_bins = int(segment_duration / 0.048)  # Adjust bin size based on your desired time bin
bin_width = 0.048 

# List to store file paths of saved figures
figure_files = []

for Unit in unit_list :

    # Unit= "Unit_37"
    spikes_in_segment = []  
    rectangles_per_segment = []
    event_frequencies = []
    
    spike_times = sorter_results[Unit]
    
    # Select the time window by slicing the sorter_results dataframe
    spike_times_window = spike_times[(spike_times >= DLC_T1) & (spike_times <= DLC_T2)]
    
    # Iterate over each segment and create a raster plot
    for i in range(num_segments) :
        start_time = DLC_T1 + i * segment_duration
        end_time = DLC_T1 + (i + 1) * segment_duration
    
        # Select events within the time window
        segment_events = spike_times_window[(spike_times_window >= start_time) & (spike_times_window <= end_time)]
    
        segment_events_relatif = segment_events - start_time
        # Compute the histogram for each segment
        hist, bin_edges = np.histogram(segment_events_relatif, bins=num_bins, range=(0, segment_duration))
        event_frequencies.append(hist)
              
        # Create rectangles for this segment
        segment_rectangles = []
    
        # Overlay colored squares for each behavior within the segment
        for behavior, color in behavior_colors.items():
            behavior_times = time_index[visual_values == behavior]
            for time in behavior_times:
                if start_time <= time <= end_time:
                    time_relative = time - start_time
                    # Include color information
                    segment_rectangles.append((time_relative, time_relative + 0.048, color))
    
        rectangles_per_segment.append(segment_rectangles)
        
    max_frequ = max(max(frequ) for frequ in event_frequencies)
        
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define the height of the rectangles
    rect_height = 1
    # Initialize a variable to keep track of the vertical offset
    vertical_offset = 0
    
    # Iterate over segments and their rectangles
    for segment in range(num_segments):
        for left, right, color in rectangles_per_segment[segment]:
            ax.fill_betweenx([vertical_offset, rect_height + vertical_offset], left, right, color=color)
        
        bin_centers = bin_edges[:-1] + bin_width / 2
        ax.bar(bin_centers, event_frequencies[segment]/max_frequ, width=bin_width, align='center', bottom=vertical_offset, color="black")
        vertical_offset += rect_height
        
    
    # Set axis limits and labels
    ax.set_xlim(0, segment_duration)
    ax.set_ylim(-rect_height/2, vertical_offset)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized spiking frequency')
    plt.title(rf"# {Unit}")
    # plt.show()
    # Save the figure to a file
    fig_name = os.path.join(save_dir, f"{Unit}_overlayBehavior.png")
    fig.savefig(fig_name)
    plt.close(fig)  # Close the figure
    
    # Append the file path to the list
    figure_files.append(fig_name)

# Define the number of rows and columns for the layout
num_rows = 4
num_cols = 2
num_plots_per_layout = num_rows * num_cols

# Calculate the number of layouts needed
num_layouts = (len(figure_files) + num_plots_per_layout - 1) // num_plots_per_layout

for layout_idx in range(num_layouts):
    start_idx = layout_idx * num_plots_per_layout
    end_idx = (layout_idx + 1) * num_plots_per_layout
    figure_files_slice = figure_files[start_idx:end_idx]

    # Create a layout to display the figures
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(30, 40))

    # Flatten the axes for easier iteration
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        file_idx = i + start_idx
        if file_idx < len(figure_files):
            # Load and display the figure from the file
            img = plt.imread(figure_files[file_idx])
            ax.axis('off')  # Turn off axis labels
            ax.imshow(img)
        else:
            # No more figures to display, hide this subplot
            ax.axis('off')

    # Save the combined layout
    # layout_name = os.path.join(save_dir, f"combined_layout_{layout_idx}.png")
    # plt.savefig(layout_name)
    plt.show()
    

#%% Create transition dataframe

Behavior = DLC[['visual']].copy()
Behavior.index.name = 'time'

# Fill "nan" values in "visual" with a consistent label, like "nan_behavior"
Behavior['visual'] = Behavior['visual'].fillna('nan_behavior')

# Step 1: Shift the "visual" column to create "next_behavior"
Behavior['next_behavior'] = Behavior['visual'].shift(-1)

# Step 2: Calculate the difference between "visual" and "next_behavior"
Behavior['transition'] = Behavior['visual'] != Behavior['next_behavior']

# Step 3: Filter the DataFrame for transitions
transitions = Behavior[Behavior['transition']]

# Drop unnecessary columns, rename previous behavior, drop last line
transitions = transitions[['visual', 'next_behavior']]
transitions = transitions.rename(columns={'visual': 'prev_behavior'})
transitions = transitions.drop(transitions.index[-1])

print(transitions)


#%% Create PSTH on transitions

# Define  PSTH parameters
time_window = 0.5  # Time start of window before center in seconds

# Identify unique behaviors
# unique_behaviors = transitions['prev_behavior'].unique()
# Order unique behaviors
unique_behaviors = ['nan_behavior', 'grooming', 'Snout rotate right', 'Snout rotate left',
       ' rotate right',' rotate left', 'stand', 'straight',
       'stand against wall', 'imobile', ]
    
for Unit in unit_list :

    spike_times = sorter_results[Unit]
    
    # Create a figure for this unit
    fig, axes = plt.subplots(5, 4, figsize=(16, 10))
    fig.suptitle(f"Unit # {Unit}")
    
    for i, behavior in enumerate(unique_behaviors):
        # Extract transitions where this behavior is "prev_behavior"
        transitions_prev = transitions[transitions['prev_behavior'] == behavior]
        
        # Extract transitions where this behavior is "next_behavior"
        transitions_next = transitions[transitions['next_behavior'] == behavior]

        # Create empty lists to store spike times
        spikes_prev, spikes_next = [], []
            
        # Loop through transitions where this behavior is "prev_behavior"
        for transition_time in transitions_prev.index:
            # Extract action potentials within the time window for prev_behavior
            spike_times_prev = spike_times[(spike_times >= transition_time - time_window) & (spike_times < transition_time + time_window)]
            # spike_times_prev_relatif = spike_times_prev - transition_time     
            spikes_prev.append(spike_times_prev - transition_time)
        
        # Loop through transitions where this behavior is "next_behavior"
        for transition_time in transitions_next.index:
            # Extract action potentials within the time window for next_behavior
            spike_times_next = spike_times[(spike_times >= transition_time - time_window) & (spike_times < transition_time + time_window)]
            spikes_next.append(spike_times_next - transition_time)
            
        
        # Add PSTH subplots
        ax1 = axes[i // 2, i % 2 * 2]  
        ax2 = axes[i // 2, i % 2 * 2 + 1]
        
        color = behavior_colors[behavior]
        
        ax1.set_ylabel('Spikes/10ms')
        histo1 = ax1.hist([elt for lst in spikes_next for elt in lst], bins=np.arange(-time_window, time_window + 0.025, 0.025))
        max_histo1 = max(histo1[0])
        ax1.add_patch(Rectangle((0, 0), time_window, max_histo1, color=color, alpha=0.2))
        
        histo2 = ax2.hist([elt for lst in spikes_prev for elt in lst], bins=np.arange(-time_window, time_window + 0.025, 0.025))
        max_histo2 = max(histo2[0])
        ax2.add_patch(Rectangle((-time_window, 0), time_window, max_histo2, color=color, alpha=0.2))
        
        # fig.suptitle(rf"# {Unit} - PSTH for {behavior} (start / end)")
        # Set labels for each subplot
        ax1.set_xlabel('Time (s)')
        ax2.set_xlabel('Time (s)')
        
        ax1.set_title(f"PSTH for {behavior} (start)")
        ax2.set_title(f"PSTH for {behavior} (end)")
        
    # Adjust the layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the layout
    transition_name = os.path.join(save_dir, f"transition_{Unit}.png")
    plt.savefig(transition_name)
    
    # Show the figure for this unit
    plt.show()

                                                                                                                                                                                                                                                                                                                                                                                           


#%% Extra

DLC['Counter-clockwise rotation'] = DLC['angle_change'] < 0 
DLC['Clockwise rotation'] = DLC['angle_change'] > 0
for index, row in DLC.iterrows():
    if row['angle_change'] < 0 and not row['IMM_ARR']:
        print("Counter-clockwise rotation")
    elif row['angle_change'] > 0 and not row['IMM_ARR']:
        print("Clockwise rotation")
    else:
        print("No rotation")

# Calculate the angle between (x_Snout, y_Snout) and (x-Tail_base, y_Tail_base) for each row
DLC['angle'] = np.arctan2(DLC['y_Snout'] - DLC['y_Tail_base'], DLC['x_Snout'] - DLC['x-Tail_base'])

# Calculate the change in angles
DLC['angle_change'] = DLC['angle'].diff()

# Create columns for rotation direction
DLC['Counter-clockwise rotation'] = (DLC['angle_change'] > 0) & (~DLC['IMM_ARR'])
DLC['Clockwise rotation'] = (DLC['angle_change'] < 0) & (~DLC['IMM_ARR'])


# Calculate the angle between (x_Snout, y_Snout) and (x-Tail_base, y_Tail_base) for each row
DLC['angle'] = np.arctan2(DLC['y_Snout'] - DLC['y_Tail_base'], DLC['x_Snout'] - DLC['x-Tail_base'])

# Calculate the change in angles
DLC['angle_change'] = DLC['angle'].diff()

# Create columns for rotation direction
DLC['Rotation'] = "No Rotation"
DLC.loc[(DLC['angle_change'] < 0) & (~DLC['IMM_ARR']), 'Rotation'] = "Counter-clockwise Rotation"
DLC.loc[(DLC['angle_change'] > 0) & (~DLC['IMM_ARR']), 'Rotation'] = "Clockwise Rotation"

# Replace True/False with "Clockwise Rotation"/"Counter-clockwise Rotation"
DLC['Rotation'] = DLC['Rotation'].replace({True: "Clockwise Rotation", False: "Counter-clockwise Rotation"})


# Iterate through the DataFrame and print messages
for index, row in DLC.iterrows():
    if row['Counter-clockwise rotation']:
        print("Counter-clockwise rotation")
    elif row['Clockwise rotation']:
        print("Clockwise rotation")
    else:
        print("No rotation")
        
        

DLC['speed_x-Tail_Base'].hist(bins=50, figsize=(16,20), xlabelsize=8, ylabelsize=8)
sns.kdeplot(data=DLC['speed_x-Tail_Base'], fill=True)

#calcule vitesse y_tail
DLC['speed_y-Tail_Base'].hist(bins=50, figsize=(16,20), xlabelsize=8, ylabelsize=8)
sns.kdeplot(data=DLC['speed_y-Tail_Base'], fill=True)

## Distribution avec .hist
DLC.iloc[:,2:].hist(figsize=(16,20), xlabelsize=8, ylabelsize=8)

## Distribution avec Seaborn
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(DLC.columns[2:]):
    ax = axes[i]
    sns.kdeplot(data=DLC[col], ax=ax, fill=True)
    ax.set_title(f"KDE Plot for {col}")
    # ax.set_xlabel("X-axis Label")
    ax.set_ylabel("Density")



