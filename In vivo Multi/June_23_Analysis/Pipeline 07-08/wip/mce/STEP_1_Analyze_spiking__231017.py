# -*- coding: utf-8 -*-
"""
Created on October 18th 2023

@author: Matilde Cordero-Erausquin

Performs spiking analysis : 
    - computes mean and max frequency for each unit (outside stimulation protocol)
    - computes autocorrelograms (outside stimulation protocol) 
    - computes ACG peak and baseline biases (according to https://doi.org/10.1093/cercor/bhx012)
    - computes ordered ACG heatmaps

Inputs : spikesorting results (spike times and inst frequency) + time of stimulation (spikeinterface)

Outputs : Excel file with spiking parameters for each unit

"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import pickle


#%%Parameters

## Dans dossier Concatenated (pour temps de stim)
session_name = '0026_29_07'
concatenated_signals_path = r'D:/ePhy/SI_Data/concatenated_signals'
signal_folder = rf'{concatenated_signals_path}/{session_name}'

## Dans dossier Spikesorting results
session_results = session_name
spikesorting_results_path = r'D:/ePhy/SI_Data/spikesorting_results'
sorter_name = "kilosort3"
sorter_folder = rf'{spikesorting_results_path}/{session_results}/{sorter_name}/curated/processing_data'
spike_times_xls = rf'{sorter_folder}/spike_times.xlsx'
file_instR = "/instantaneous_rates_20ms.xlsx"

output_xls = rf'{sorter_folder}/spiking.xlsx'

sampling_rate = 20000#Hz

# plots_path = r'D:\ePhy\SI_Data\plots'

# #Window for raster
# window_before = 200 #ms before the stimulation
# window_after = 200 #ms before the stimulation
# window_first_spike = 10 #ms after stimulation
# stim_duration = 100 #ms

# #Reliability windows
# window_reliability=40#ms

check_waveform = False



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

# def Check_Save_Dir(save_path):
#     """
#     Check if the save folder exists. If not, create it.

#     Args:
#         save_path (str): Path to the save folder.

#     """
#     import os
#     isExist = os.path.exists(save_path)
#     if not isExist:
#         os.makedirs(save_path)  # Create folder for the experiment if it does not already exist

#     return


# def exponential_decay(x, A, B, tau, C):
#     return A * np.exp(-(x-B) / tau) + C


#%%Loading

#Load units
recordings_info = Get_recordings_info(session_name,concatenated_signals_path,sorter_folder)

#Load Stim ttl times
stim_idx = recordings_info['stim_ttl_on'][0::2]

#Time window to be outside of stimulation protocol
T1 = min(stim_idx) / sampling_rate
T2 = (max(stim_idx) / sampling_rate) + 5

print(rf"Loading spikesorting results for session {session_name}")
sorter_results = pd.read_excel(spike_times_xls)
sorter_results = sorter_results.drop(sorter_results.columns[0], axis=1)
# we = si.load_waveforms(rf'{sorter_folder}/curated/waveforms')

#Get rid of spikes during stimulation protocol
for column in sorter_results.columns:
    # Create a mask for values outside the time range
    mask = (sorter_results[column] >= T1) & (sorter_results[column] <= T2)
    # Update the column with NaN for values inside the range
    sorter_results.loc[mask, column] = np.nan
sorter_results = sorter_results.dropna(axis=0, how='all')

unit_list = sorter_results.columns
# unit_list = sorter_results.columns.str.slice(5,)
print(unit_list)

##Load instantaneous frequencies
frequfilename = sorter_folder + file_instR
instR = pd.read_excel(frequfilename, decimal=',')
instR.set_index(instR.columns[0], inplace = True)
instR.index = instR.index/10

#Get rid of instR during stimulation protocol
mask = (instR.index >= T1) & (instR.index <= T2)
instR = instR[~mask]


#%%Analyzing


## Frequ moy et max

mean_firing = instR.mean(axis=0)
max_firing = instR.max(axis=0)
            
#%% All clusters correlogramm

bsl_ACG, peak_ACG, unit_plots = [], [], []

for unit in unit_list:
    
    spike_times = sorter_results[unit]*1000
    
    # Paramètres pour l'autocorrelogramme
    bin_size = 1  # Taille des intervalles de temps
    max_lag = 250  # Durée maximale de la corrélation

    # Calcul des différences entre les temps d'événements consécutifs
    event_diffs = np.diff(spike_times)
    
    symetric_event_diffs = np.hstack((-event_diffs,event_diffs))
    
    # Création des bins pour l'autocorrelogramme
    bins = np.arange(-max_lag, max_lag + bin_size, bin_size)
    
    # Calcul de l'autocorrelogramme
    autocorrelogram, _ = np.histogram(symetric_event_diffs, bins=bins)
    
    # Affichage de l'autocorrelogramme
    plt.figure()
    plt.bar(bins[:-1], autocorrelogram, width=bin_size, align='edge')
    plt.xlabel('time in ms')
    plt.ylabel('Events')
    plt.title(rf'Unit # {unit}')
    plt.show()
    
    ################# Calculating ACG bias ######################
    
    # Create the x-axis values
    x_values = list(range(-max_lag, max_lag, bin_size))

    # Create a DataFrame with x-axis as the index and correlogram values as a single column
    ACG = pd.DataFrame({"autocorrelogram": autocorrelogram}, index=x_values)

    # # Calculate moving average to smooth the ACG
    # window_size = 5
    # ACG['Moving Average'] = ACG['autocorrelogram'].rolling(window=window_size).mean()
    
    # Apply a Gaussian filter to the autocorrelogram
    ACG['gaussian'] = gaussian_filter1d(ACG['autocorrelogram'], sigma=2)  # Adjust sigma as needed
    
    # Paramètres pour analyse Peak et baseline ACG
    mask_peak = (ACG.index > 0) & (ACG.index < 50)
    mask_bsl = (ACG.index > 50)
    
    ACG['peak_cum'] = ACG['gaussian'][mask_peak].cumsum()
      
    peak_sum = ACG['peak_cum'].max(skipna=True)
    peak_ACG_bias = (ACG['peak_cum'] >= peak_sum/2).idxmax()

    ACG['bsl_cum'] = ACG['gaussian'][mask_bsl].cumsum()
    
    bsl_sum = ACG['bsl_cum'].max(skipna=True)
    bsl_ACG_bias = (ACG['bsl_cum'] >= bsl_sum/2).idxmax()
             
    # print(peak_ACG_bias, bsl_ACG_bias)
    bsl_ACG.append(bsl_ACG_bias)
    peak_ACG.append(peak_ACG_bias)
    
    # plt.savefig(rf'\\equipe2-nas1\Gilles.DELBECQ\Data\ePhy\July_23\test\correlo_record1_{index+1}.png')
   
    
    ################# Preparing heatmaps ######################
               
    gaussian_max = max(ACG['gaussian'])
    unit_plots.append(ACG['gaussian']/gaussian_max)

# Get the sorted indices based on peak_ACG values
sorted_indices = np.argsort(peak_ACG)

# Rearrange the imshow_plots and peak_ACG_values based on sorted indices
sorted_plots = [unit_plots[i] for i in sorted_indices]

stacked_plot = np.vstack(sorted_plots)    
plt.figure()
plt.imshow(stacked_plot, cmap='jet', aspect='auto')
plt.xlabel('Time (ms)')
plt.ylabel('Units')

# Show the color bar
cbar = plt.colorbar()
cbar.set_label('Intensity')

# Show the plot
plt.show()
    
    

#%%Saving

"""
Saving in dataframes
"""
spiking = pd.DataFrame({
    'meanF' : mean_firing,
    'maxF' : max_firing,
    'peak_ACG_bias' : peak_ACG,
    'bsl_ACG_bias' : bsl_ACG
    }, index=unit_list)

spiking.to_excel(output_xls)
