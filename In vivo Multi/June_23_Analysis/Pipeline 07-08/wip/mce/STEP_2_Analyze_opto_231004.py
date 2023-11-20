# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 11:32:27 2023

@author: Gilles Delbecq

Performs optotag analysis : 
    - compute delay and jitter for each unit (in the window response of 10ms and beyond)
    - compute reliability of the photostimulation for each unit
    - fit decay time of spiking during the window of stimulation

Inputs : spikesorting results (spikeinterface)

Outputs : Excel files with parameters for each units

"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
import math

# import spikeinterface as si
# import spikeinterface.sorters as ss
# import spikeinterface.postprocessing as spost
# import spikeinterface.widgets as sw

# from neo.core import SpikeTrain
# from quantities import ms, s, Hz

# from elephant.statistics import time_histogram, instantaneous_rate

# from elephant import kernels

import pickle
import time

import seaborn as sns
import scipy.cluster.hierarchy as sch

#%%Parameters
session_name = '0026_01_08'
session_results = session_name

spikesorting_results_path = r'D:/ePhy/SI_Data/spikesorting_results'
concatenated_signals_path = r'D:/ePhy/SI_Data/concatenated_signals'

sampling_rate = 20000#Hz

sorter_name = "kilosort3"
sorter_folder = rf'{spikesorting_results_path}/{session_results}/{sorter_name}/curated/processing_data'
signal_folder = rf'{concatenated_signals_path}/{session_name}'
spike_times_xls = rf'{sorter_folder}/spike_times.xlsx'
output_xls = rf'{sorter_folder}/optotag_infos_fit.xlsx'

plots_path = r'D:\ePhy\SI_Data\plots'

#Window for raster
window_before = 200 #ms before the stimulation
window_after = 200 #ms before the stimulation
window_first_spike = 10 #ms after stimulation
stim_duration = 100 #ms

#Reliability windows
window_reliability=40#ms

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


def exponential_decay(x, A, B, tau, C):
    return A * np.exp(-(x-B) / tau) + C


#%%Loading
#Load units
recordings_info = Get_recordings_info(session_name,concatenated_signals_path,sorter_folder)

#Load Stim ttl times
stim_idx = recordings_info['stim_ttl_on'][0::2]

# stim_idx = stim_idx - 7777024


#Select values of stim idx spaced of 2000 pts (100ms) = optotag stim
#Do that to get rid of other stim (for perturbation for instance)

diffs = np.diff(stim_idx)

mask = np.logical_and(diffs >= 19984 - 200, diffs <= 19984 + 200)


selected_optotag_stim_idx = stim_idx[1:][mask]

selected_optotag_stim_times = selected_optotag_stim_idx/sampling_rate#secondes

nb_stim = len(selected_optotag_stim_times)
print('Nombre de stimulations opto :')
print(nb_stim)


print(rf"Loading spikesorting results for session {session_name}")
sorter_results = pd.read_excel(spike_times_xls)
# we = si.load_waveforms(rf'{sorter_folder}/curated/waveforms')

# signal = si.load_extractor(signal_folder)

unit_list = sorter_results.columns
# unit_list = sorter_results.columns.str.slice(5,)
print(unit_list)
#%% Optotagging


successes,delays,jitters,z_scores, delays_w, jitters_w, A, B, tau, C = [],[],[],[],[],[],[],[],[],[]

for unit in unit_list[1:] :

    # unit= "Unit_22"
    
    spike_times = sorter_results[unit]
    
    spikes_in_window = []    
    first_spikes = []
    
    """
    Slicing
    """
    
    for stimulation in selected_optotag_stim_times:
        start_time = stimulation - window_before/1000
        end_time = stimulation + window_after/1000
        
        selected_events = spike_times[(spike_times >= start_time) & (spike_times <= end_time)]
        selected_events_relatif = selected_events-stimulation
        
        spikes_in_window.append(selected_events_relatif)        
        
    """
    Raster plot and overlaid PSTH
    """

    fig, ax1 = plt.subplots()
    histo = ax1.hist([elt for lst in spikes_in_window for elt in lst], bins=np.arange(-0.2, 0.21, 0.01), color = "grey", alpha=0.2)
    ax1.set_ylabel('Spikes/10ms')

    histo_val = histo[0]
    histo_bin = histo[1]
    
    ax2 = ax1.twinx()
    ax2.eventplot(spikes_in_window, color='black')
    ax2.set_ylabel('Stimulation #', rotation=-90, labelpad=20)

    plt.axvspan(0, 0.100, color='blue', alpha=0.05)
    plt.xlim(-0.20, 0.200)
    plt.xlabel('Time (s)')
    plt.title(rf"Unit # {unit}")
    
    # savefig_path = rf'{sorter_folder}/curated/processing_data/plots/rasterplot_opto/unit_{unit}.png'
    # Check_Save_Dir(os.path.dirname(savefig_path))
    # plt.savefig(savefig_path)
    
    
    
    """
    Fit exponential during full stimulation window
    """
    
    # Extract the values from the DataFrame
    xdata = histo_bin[20:30]
    ydata = histo_val[20:30]
    
    ###  condition to check if neuron spikes at all during window
    # sum(ydata) == 0:
    #     A_fit, B_fit, tau_fit, C_fit = [np.nan, np.nan, np.nan, np.nan]
    try:
        # Set bounds for each parameter
        bounds = ([0, 0, 0, min(histo_val)], [10*max(histo_val), 0.05, np.inf, max(histo_val)])
        
        # Perform the fit
        params, covariance = curve_fit(exponential_decay, xdata, ydata, bounds=bounds, p0=(ydata[0], 0, 0.05, ydata[9]))
        
        # Extract the fitting parameters
        A_fit, B_fit, tau_fit, C_fit = params
        
        # Calculate the fitted curve
        y_fit = exponential_decay(xdata, A_fit, B_fit, tau_fit, C_fit)
    
        # plt.figure(figsize=(8, 6))
        # plt.scatter(xdata, ydata, label='Data')
        ax1.plot(xdata, y_fit, 'r', label='Fitted Curve')
        # plt.xlabel('Time')
        # plt.ylabel('Value')
        # plt.title('Exponential Decay Fit')
        # plt.legend()
    
    except ValueError as e:
        error_message = str(e)
        A_fit, B_fit, tau_fit, C_fit = [np.nan, np.nan, np.nan, np.nan]
    except RuntimeError as e:
        error_message = str(e)
        A_fit, B_fit, tau_fit, C_fit = [np.nan, np.nan, np.nan, np.nan]
        
    decay_time = tau_fit
    print(f"Decay Time (tau) {unit}: {decay_time:.6f}")
    
    
        
    """
    Compute
        - SD from histo
        - Z-score of response window
    """

    mean_histo = np.mean(histo_val[0:19])                ## si 40 bins
    SD_histo = np.std(histo_val[0:19])

    # plt.figure()
    # plt.hist(histo_val[0:19])
    # plt.axvspan(mean_histo-2*SD_histo, mean_histo+2*SD_histo, color='green', alpha=0.1)
    # plt.axvline(histo_val[20], color='red', alpha=1)

    # plt.title(rf"Unit # {unit}, Count after stim vs. 100 ms baseline ")
    # plt.xlabel('Nb spikes/10 ms')
    # plt.ylabel('Count')
    
    if mean_histo == 0:
        z_score = A_fit
    else:
        # Calculate the z-score for the single bin in the response window
        z_score = (histo_val[20] - mean_histo) / SD_histo
    
    """
    Compute
        - first spike
        - % Success
    """

    for stim in spikes_in_window:
        first_spike = next((x for x in stim if (x > 0)), np.nan)
        first_spikes.append(first_spike)
    
    first_spikes_w = [x for x in first_spikes if (not math.isnan(x) and x < window_first_spike/1000)]
    
    # sorted_spikes = sorted(first_spikes, key=lambda x: (np.isnan(x), x))
    # # Create the scatter plot
    # plt.scatter(range(1, len(sorted_spikes) + 1), sorted_spikes)
                
    ### Success = trial where there is a spike in window_first_spike ###

    nb_succes = len(first_spikes_w)
    tx_succes = nb_succes*100 /nb_stim

    mean_first_spike = 1000*np.nanmean(first_spikes)
    mean_first_spike_w = 1000*np.nanmean(first_spikes_w)
    
    jitter = 1000*np.nanstd(first_spikes)
    jitter_w = 1000*np.nanstd(first_spikes_w)

    text_line1 = 'z-score= ' + "%.2f" % z_score + ', success rate= ' + "%.2f" % tx_succes +'% of trials\n' + 'delay first spike= ' + "%.2f" % mean_first_spike + '+-' + "%.2f" % jitter + 'ms, delay in window=' + "%.2f" % mean_first_spike_w + '+-' + "%.2f" % jitter_w + 'ms'
    plt.text(-0.2, -80, text_line1)
    plt.show()
    
    z_scores.append(z_score)         
    delays.append(mean_first_spike)
    jitters.append(jitter)
    delays_w.append(mean_first_spike_w)
    jitters_w.append(jitter_w)
    successes.append(tx_succes)
    A.append(A_fit)
    B.append(B_fit)
    tau.append(tau_fit)
    C.append(C_fit)
    
            

"""
Saving in dataframes
"""
df_optotag = pd.DataFrame({
    'Z-score' : z_scores,
    '% success' : successes,
    'delays' : delays,
    'jitters' : jitters,   
    'delays_w' : delays_w,
    'jitters_w' : jitters_w,   
    'A_fit' : A,
    'B_fit' : B,
    'tau_fit' : tau,
    'C_fit' : C
    },    index = unit_list[1:])

df_optotag.to_excel(output_xls)
