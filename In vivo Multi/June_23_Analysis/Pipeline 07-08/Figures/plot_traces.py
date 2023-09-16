# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 11:30:59 2023

plots raw traces from spikeinterface concatenated signals


@author: Gilles Delbecq
"""
#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import spikeinterface as si
import spikeinterface.sorters as ss
import pickle

#%%Parameters
session_name = '0022_01_08'
chans_to_plot = [2]

time_window = [33.4,33.6] #s


spikesorting_results_path = r"D:\ePhy\SI_Data\spikesorting_results"
concatenated_signals_path = r'D:\ePhy\SI_Data\concatenated_signals'
plots_path = r'D:\ePhy\SI_Data\plots\signals'

sorter_name = "kilosort3"

sorter_folder = rf'{spikesorting_results_path}/{session_name}/{sorter_name}'
signal_folder = rf'{concatenated_signals_path}/{session_name}'

sampling_rate = 20000 #Hz

#%% Functions

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


#%% Loadings
"""
Load units
"""
recordings_info = Get_recordings_info(session_name,concatenated_signals_path,spikesorting_results_path)

print(rf"Loading spikesorting results for session {session_name}")
sorter_results = ss.NpzSortingExtractor.load_from_folder(rf'{sorter_folder}/curated').remove_empty_units()
signal = si.load_extractor(signal_folder)
we = si.load_waveforms(rf'{sorter_folder}/curated/waveforms')

time_axis = signal.get_times()
unit_list = sorter_results.get_unit_ids()

print(rf"{len(sorter_results.get_unit_ids())} units loaded")

#%% Plot the traces

recording = signal


# Convertissez les temps de début et de fin en indices d'échantillon
t_start_sample = int(time_window[0] * recording.get_sampling_frequency())
t_stop_sample = int(time_window[1] * recording.get_sampling_frequency())



plt.figure(figsize=(10, 8))

# Tracez chaque signal
for i, channel in enumerate(chans_to_plot, 1):
    # Extrayez le segment de signal pour l'intervalle spécifié
    segment_data = recording.get_traces(channel_ids=[str(channel)], start_frame=t_start_sample, end_frame=t_stop_sample)
    
    time_axis = np.linspace(time_window[0], time_window[1], len(segment_data))
    
    # Utilisez l'argument sharex pour partager le même axe x
    if i == 1:
        ax1 = plt.subplot(len(chans_to_plot), 1, i)
        plt.plot(time_axis, segment_data)
    else:
        plt.subplot(len(chans_to_plot), 1, i, sharex=ax1)
        plt.plot(time_axis, segment_data)
    
    plt.title(f'Channel {channel}')
    
    # Si ce n'est pas le dernier subplot, retirez les étiquettes de l'axe x
    if i != len(chans_to_plot):
        plt.xlabel('')
    else:
        plt.xlabel('Time (s)')
    
    plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

plt.ylim(-2500,2000)

Check_Save_Dir(plots_path)
plt.savefig(rf"{plots_path}/signal_zoom.svg")

#%% Plot spikes on trace
import spikeinterface.widgets as sw

#TODO : saving and beautification
sw.plot_spikes_on_traces(we,order_channel_by_depth=True)
sw.plot_spikes_on_traces(we,channel_ids=['12','13','15','14'],order_channel_by_depth=True)
