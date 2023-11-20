# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 11:32:27 2023

@author: Gilles Delbecq

Performs optotag analysis : 
    - compute delay and jitter for each unit
    - compute reliability of the photostimulation for each unit

Inputs : spikesorting results (spikeinterface)

Outputs : Excel files with parameters for each units

"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import spikeinterface as si
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.widgets as sw

from neo.core import SpikeTrain
from quantities import ms, s, Hz

from elephant.statistics import time_histogram, instantaneous_rate

from elephant import kernels

import pickle
import time

import seaborn as sns
import scipy.cluster.hierarchy as sch

#%%Parameters
session_name = '0022_01_08'
mocap_session = "01"

spikesorting_results_path = r"D:\ePhy\SI_Data\spikesorting_results"
concatenated_signals_path = r'D:\ePhy\SI_Data\concatenated_signals'

sampling_rate = 20000#Hz

sorter_name = "kilosort3"
sorter_folder = rf'{spikesorting_results_path}/{session_name}/{sorter_name}'
signal_folder = rf'{concatenated_signals_path}/{session_name}'

plots_path = r'D:\ePhy\SI_Data\plots'

#Window for raster
window_before = 200 #ms before the stimulation
window_after = 400 #ms before the stimulation
stim_duration = 100 #ms

#Reliability windows
window_reliability=40#ms

check_waveform = False



#%%Functions
def Get_recordings_info(session_name, concatenated_signals_path, spikesorting_results_path):
    """
    Cette fonction récupère les informations d'enregistrement à partir d'un fichier de métadonnées
    dans le dossier de signaux concaténés.

    Args:
        session_name (str): Le nom de la session d'enregistrement.
        concatenated_signals_path (str): Le chemin vers le dossier contenant les signaux concaténés.
        spikesorting_results_path (str): Le chemin vers le dossier des résultats du tri des spikes.

    Returns:
        dict or None: Un dictionnaire contenant les métadonnées si la lecture est réussie,
        ou None si la lecture échoue.

    Raises:
        Exception: Si une erreur se produit pendant la lecture du fichier.

    """
    try:
        # Construire le chemin complet vers le fichier de métadonnées
        path = rf'{concatenated_signals_path}/{session_name}/'
        
        # Lire le fichier de métadonnées à l'aide de la bibliothèque pickle
        print("Lecture du fichier ttl_idx dans le dossier Intan...")
        metadata = pickle.load(open(rf"{path}/ttl_idx.pickle", "rb"))
        
    except Exception as e:
        # Gérer toute exception qui pourrait se produire pendant la lecture du fichier
        print("Aucune information d'enregistrement trouvée dans le dossier Intan. Veuillez exécuter l'étape 0.")
        metadata = None  # Aucune métadonnée disponible en cas d'erreur
    
    print('Terminé')
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




#%%Loading
#Load units
recordings_info = Get_recordings_info(session_name,concatenated_signals_path,spikesorting_results_path)

#Load Stim ttl times
stim_idx = recordings_info['stim_ttl_on'][92:][0::2]



#Select values of stim idx spaced of 2000 pts (100ms) = optotag stim
#Do that to get rid of other stim (for perturbation for instance)

diffs = np.diff(stim_idx)

mask = np.logical_and(diffs >= 19984 - 200, diffs <= 19984 + 200)


selected_optotag_stim_idx = stim_idx[1:][mask]

selected_optotag_stim_times = selected_optotag_stim_idx/sampling_rate#secondes




print(rf"Loading spikesorting results for session {session_name}")
sorter_results = ss.NpzSortingExtractor.load_from_folder(rf'{sorter_folder}/curated').remove_empty_units()
we = si.load_waveforms(rf'{sorter_folder}/curated/waveforms')

signal = si.load_extractor(signal_folder)


unit_list = sorter_results.get_unit_ids()

#%% Optotagging

reliability_scores,delays,jitters = [],[],[]

for unit in unit_list:
    spike_idx = sorter_results.get_unit_spike_train(unit_id=unit)
    
    spikes_in_window = []
    
    """
    Slicing
    """
    
    for stimulation in selected_optotag_stim_idx:
        start_idx = int(stimulation - ((window_before/1000)*sampling_rate))
        end_idx  = int(stimulation + ((window_after/1000)*sampling_rate))
        
        selected_events = spike_idx[(spike_idx >= start_idx) & (spike_idx <= end_idx)]
        
        spikes_in_window.append(((selected_events-stimulation)/sampling_rate)*1000)
        
        
    """
    Raster plot    
    """
    plt.figure()
    plt.eventplot(spikes_in_window, color='black')
    plt.title(rf"Unit # {unit}")
    plt.xlabel('Time (ms)')
    plt.ylabel('Stimulation #')
    plt.axvspan(0, 100, color='blue', alpha=0.05)
    plt.xlim(-100, 400)
    
    savefig_path = rf'{sorter_folder}/curated/processing_data/plots/rasterplot_opto/unit_{unit}.png'
    Check_Save_Dir(os.path.dirname(savefig_path))
    plt.savefig(savefig_path)
    savefig_path = rf'{sorter_folder}/curated/processing_data/plots/rasterplot_opto/unit_{unit}.svg'
    Check_Save_Dir(os.path.dirname(savefig_path))
    plt.savefig(savefig_path)
    
    plt.close()
    
    plt.figure()
    plt.hist(np.concatenate(spikes_in_window), bins=np.arange(-100, 400, 5), color='black')  # Ajustez les bacs et la plage selon vos besoins
    plt.title(rf"Histogram of First Spike Delays for Unit # {unit}")
    plt.xlabel('Delay (ms)')
    plt.ylabel('Count')
    plt.axvspan(0, 100, color='blue', alpha=0.05)
    plt.show()  # Ajoutez cette ligne si vous souhaitez voir l'histogramme immédiatement

    savefig_path = rf'{sorter_folder}/curated/processing_data/plots/histo_opto_opto/unit_{unit}.png'
    Check_Save_Dir(os.path.dirname(savefig_path))
    plt.savefig(savefig_path)
    
    savefig_path = rf'{sorter_folder}/curated/processing_data/plots/histo_opto_opto/unit_{unit}.svg'
    Check_Save_Dir(os.path.dirname(savefig_path))
    plt.savefig(savefig_path)

    
    plt.close()
    
    
    """
    Compute
        - first spike
        - reliability
    """
    
    first_spikes = []
    reliability = []
    for stim in spikes_in_window:
        first_spike = next((x for x in stim if x > 0), np.nan)
        first_spikes.append(first_spike)
        
        reliability_before = stim[(stim >= float(-window_reliability)) & (stim <= 0)]
        reliability_after = stim[(stim <= float(window_reliability))& (stim >= 0)]

        
        if len(reliability_before) < len(reliability_after):
            reliability.append(1)
        else:
            reliability.append(0)
        
    
    mean_first_spike = np.nanmean(first_spikes)
    jitter = np.nanstd(first_spikes)
    
    count_zeros = len(reliability) - np.count_nonzero(reliability)
    reliability_ratio = (1-(count_zeros/len(spikes_in_window)))*100
    
    reliability_scores.append(reliability_ratio)
    delays.append(mean_first_spike)
    jitters.append(jitter)
    
    print(rf'Unit # {unit} delay = {round(mean_first_spike,2)} ms +/- {round(jitter,2)} ms and reliability of {round(reliability_ratio,2)}%')

    if check_waveform == True:
        for i in selected_optotag_stim_times[0:10]:
            stim_time = i/sampling_rate
            sw.plot_spikes_on_traces(we,time_range=[stim_time-1,stim_time+1])
            # plt.axvline(1.8)
            plt.axvspan(stim_time, stim_time+0.1, alpha=0.3)
            

# Créer un histogramme
plt.figure()
plt.hist(reliability_scores, bins=10, edgecolor='black')  # Vous pouvez ajuster le nombre de bacs (bins) selon vos besoins

# Ajouter des labels aux axes et un titre au graphique
plt.xlabel('Relibility (%)')
plt.ylabel('Count')
plt.title(rf'Reliability session {session_name}')


# Afficher l'histogramme
plt.show()

savefig_path = rf'{sorter_folder}/curated/processing_data/plots/reliability_histo.png'
Check_Save_Dir(os.path.dirname(savefig_path))
plt.savefig(savefig_path)


"""
Plot reliability
"""
plt.figure()
plt.scatter(delays, reliability_scores, s=jitters, c='blue', alpha=0.3)

for i, name in enumerate(unit_list):
    plt.text(delays[i], reliability_scores[i], name, fontsize=11, ha='center', va='bottom')


plt.xlabel('Delay (ms)')
plt.ylabel('Reliability (%)')
plt.legend(['Jitter (ms)'])

savefig_path = rf'{sorter_folder}/curated/processing_data/plots/reliability_scatter.png'
Check_Save_Dir(os.path.dirname(savefig_path))
plt.savefig(savefig_path)


"""
Saving in dataframes
"""
df_optotag = pd.DataFrame({
    'units' : unit_list,
    'reliability_scores' : reliability_scores,
    'delays' : delays,
    'jitters' : jitters   
    })

df_optotag.to_excel(rf'{sorter_folder}/curated/processing_data/optotag_infos.xlsx')
