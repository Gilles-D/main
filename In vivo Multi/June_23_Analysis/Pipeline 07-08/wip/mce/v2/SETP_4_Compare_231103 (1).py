# -*- coding: utf-8 -*-
"""
Spyder Editor
@author: Matilde Cordero-Erausquin

Compare units from different sessions 

Inputs : 
    -  units_data = excel file with parameters for each unit

"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

import PySimpleGUI as sg
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#%%Parameters
Sessions = ['0022_28_07', '0022_31_07', '0022_01_08']

spikesorting_path = r'//EQUIPE2-NAS1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/'
sorter_name = "/kilosort3"
data_folder = "/curated/processing_data"

input_xls = '/units_data.xlsx'


#%% Load units_data

datas = []

for Session in Sessions:
    
    units_data_filename = spikesorting_path + Session + sorter_name + data_folder + "/units_data.xlsx"
    data = pd.read_excel(units_data_filename, decimal=",")
    datas.append(data)

# Concatenate the list of dataframes into a single dataframe
sessions_data = pd.concat(datas, ignore_index=True)
sessions_data = sessions_data.rename(columns={sessions_data.columns[0]: "Units"})

 

#%% GUI for exploration

shape_column = 'OPTO'   
color_palette = sns.color_palette('husl', n_colors=len(Sessions))
marker_styles = ['d', 'X'] 
  
# Create a scatterplot function for the 2x2 layout
def generate_2x2_plot(x_column, y_column, enable_annotations=False):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))    
      
    # Loop to create Plots 1 to 3
    for i, session_name in enumerate(Sessions):
        ax = axes[i // 2, i % 2]
        data_session = sessions_data[sessions_data['animal'] == session_name]
        sns.scatterplot(data=data_session, x=x_column, y=y_column, hue='animal', style=shape_column, markers=marker_styles, s=100, ax=ax, palette=[color_palette[i]], legend=False)
        if enable_annotations:
            for unit_name, x, y in zip(data_session['Units'], data_session[x_column], data_session[y_column]):
                truncated_index = str(unit_name)[5:]  # Remove the first 5 characters
                ax.text(x, y, str(truncated_index), fontsize=12, ha='center', va='bottom')

        ax.set_title(f'Data from {session_name}')
    
    # Plot 4: Data from All Sessions
    ax = axes[1, 1]
    sns.scatterplot(data=sessions_data, x=x_column, y=y_column, hue='animal', style=shape_column, markers=marker_styles, s=100, ax=ax, legend=False)
    ax.set_title('Data from All Sessions')
    
    # Adjust the layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()        

def create_layout():
    return [
        [sg.Text("Select X-axis column:"), sg.Combo(values=list(sessions_data.columns), key='x_col')],
        [sg.Text("Select Y-axis column:"), sg.Combo(values=list(sessions_data.columns), key='y_col')],
        [sg.Text("Show Unit's #"), sg.Checkbox('', key='enable_annotations')],
        [sg.Button("Generate All Plots")],
        [sg.VSeperator()],
        [sg.Canvas(key='canvas')]
    ]

layout = create_layout()
window = sg.Window("Scatterplot Generator", layout, resizable=True)
fig, axs = None, None

while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED:
        break
        
    if event == "Generate All Plots":
        x_col = values['x_col']
        y_col = values['y_col']
        enable_annotations = values['enable_annotations']  # Checkbox state        
        generate_2x2_plot(x_col, y_col, enable_annotations)
        
            
window.close()


#%% GUI for units comparison

dic_units = {
    Sessions[0]: 'Unit_52',
    Sessions[1]: 'Unit_59',
    Sessions[2]: 'Unit_65'} 

waveforms = []

# Load the waveforms
for session, unit in dic_units.items():
    unit_wf_name = unit + '_wf.xlsx'
    units_data_filename = spikesorting_path + session + sorter_name + data_folder + "/waveforms/" + unit_wf_name
    wf = pd.read_excel(units_data_filename, decimal=",")
    waveforms.append(wf)

# Create a figure with 16 subplots
fig, axs = plt.subplots(4, 4, figsize=(12, 12), sharex=True, sharey=True)
fig.suptitle("Overlay of the 3 units")

# Flatten the axs array for easier iteration
axs = axs.ravel()

# Iterate over the 16 subplots (sites)
for i in range(16):
    ax = axs[i]
    ax.set_title(f"Site {i}")  # Set the title for each subplot

    # Plot action potentials from each dataframe with different colors
    for j, dataframe in enumerate(waveforms):
        ax.plot(dataframe[i], label=f"AP{j + 1}")

    ax.legend()  # Show legends for different action potentials

plt.tight_layout()
plt.show()

















