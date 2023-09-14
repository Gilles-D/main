# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 11:30:59 2023

@author: MOCAP
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import spikeinterface as si
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost

from neo.core import SpikeTrain
from quantities import ms, s, Hz

from elephant.statistics import time_histogram, instantaneous_rate,mean_firing_rate

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
plots_path = r'D:\ePhy\SI_Data\plots'

sorter_name = "kilosort3"

sorter_folder = rf'{spikesorting_results_path}/{session_name}/{sorter_name}'
signal_folder = rf'{concatenated_signals_path}/{session_name}'

sampling_rate = 20000 #Hz

chans_to_plot = [1,2,3,4,5]


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


def list_recording_files(path, session):
    """
    List all CSV files containing the specified session in the name
    in the specified directory and its subdirectories.

    Parameters:
        path (str): The directory path to search for CSV files.
        session (str): The session to search for in the file names.

    Returns:
        list: A list of paths to CSV files containing the session in their name.
    """
    import os
    
    session = rf"_{session}_"

    csv_files = []
    for folderpath, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.lower().endswith(".xlsx") and session in filename and "Analysis" in filename:
                csv_files.append(os.path.join(folderpath, filename))

    return csv_files

def find_file_with_string(file_paths, target_string):
    """
    Find a file containing the target string in its content among the list of file paths.

    Parameters:
        file_paths (list): List of file paths to search.
        target_string (str): The target string to search for.

    Returns:
        str or None: The path of the first file containing the target string, or None if not found.
    """
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            file_content = file.read()
            if target_string in file_content:
                return file_path
    return None

def moving_average(data, window_size):
    """
    Calcule la moyenne mobile d'un ensemble de données en utilisant une fenêtre donnée.

    Args:
        data (numpy.ndarray): Les données d'entrée sur lesquelles la moyenne mobile sera calculée.
        window_size (int): La taille de la fenêtre utilisée pour la moyenne mobile.

    Returns:
        numpy.ndarray: Les données lissées après l'application de la moyenne mobile.

    """
    # Créer une fenêtre de moyenne mobile avec des coefficients égaux
    window = np.ones(window_size) / window_size

    # Appliquer la moyenne mobile en utilisant la convolution avec le mode 'same'
    smoothed_data = np.convolve(data, window, mode='same')

    return smoothed_data

def calculate_instantaneous_frequencies(event_times, total_time, window_size, step_size=1):
    """
    Calculate instantaneous frequencies of events along a fixed time axis using sliding windows.
    
    Parameters:
        event_times (numpy.ndarray): 1D array containing event times.
        total_time (float): Total duration of the recording in seconds.
        window_size (float): Size of the sliding window in seconds.
        step_size (float): Step size for sliding the window (default is 1 second).
        
    Returns:
        numpy.ndarray: Array containing instantaneous frequencies for each time point.
    """
    time_axis = np.arange(0, total_time, step_size)
    instantaneous_frequencies = np.zeros(len(time_axis))
    
    for i, t in enumerate(time_axis):
        window_start = t
        window_end = t + window_size
        events_in_window = np.sum((event_times >= window_start) & (event_times < window_end))
        frequency = events_in_window / window_size
        instantaneous_frequencies[i] = frequency
    
    return instantaneous_frequencies


def plot_waveform(session_name, sorter_folder, sites_location, unit, save=True):
    import glob
    file_pattern = rf"Unit_{unit} *"
    matching_files = glob.glob(rf"{sorter_folder}/we/curated/waveforms/{file_pattern}")
    
    print(rf"{sorter_folder}/curated/waveforms/{file_pattern}")
    
    if len(matching_files) > 0:
        print("Matching file(s) found:")
        for file_path in matching_files:
            print(file_path)
            
            df = pd.read_excel(file_path)
            max_wave = df.abs().max().max()  # Calculate the maximum value in all channels
            
            fig1 = plt.figure(figsize=(10, 12))
            ax1 = fig1.add_subplot(111)
            
            fig1.suptitle(rf'Average Waveform Unit # {unit}')
            ax1.set_xlabel('Probe location (micrometers)')
            ax1.set_ylabel('Probe location (micrometers)')
            
            for loc, prob_loc in enumerate(sites_location):
                x_offset, y_offset = prob_loc[0], prob_loc[1]
                base_x = np.linspace(-15, 15, num=len(df.iloc[:, loc]))  # Basic x-array for plot, centered
                # clust_color = 'C{}'.format(cluster)
                
                wave = df.iloc[:, loc] * 100 + max_wave * y_offset  # Adjust y_offset with the max_wave value
                ax1.plot(base_x + 2 * x_offset, wave)
                # ax1.fill_between(base_x + 2 * x_offset, wave - wf_rms[cluster + delta], wave + wf_rms[cluster + delta], alpha=wf_alpha)
                
            plt.show()
            if save == True:
                save_path = rf"{spikesorting_results_path}/{session_name}/plots/waveforms/"
                Check_Save_Dir(save_path)
                plt.savefig(rf"{save_path}/Units {unit}.png")
    else:
        print("No matching file found")
        
    return


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

#%% Show some traces

# Assumons que vous avez déjà votre objet `BinaryFolderRecording` nommé signal.
# Si non, chargez-le comme ceci (remplacez 'YOUR_PATH' par le chemin approprié):
# reader = neo.io.BinarySignalIO(filename='YOUR_PATH')
# signal = reader.read()

recording = signal
# Supposons que votre BinaryFolderRecording est déjà chargé comme ceci :
# recording = si.load_from_folder(folder_path='YOUR_FOLDER_PATH')



# Convertissez les temps de début et de fin en indices d'échantillon
t_start_sample = int(30 * recording.get_sampling_frequency())  # 30s * fréquence d'échantillonnage
t_stop_sample = int(40 * recording.get_sampling_frequency())   # 40s * fréquence d'échantillonnage

plt.figure(figsize=(10, 8))

# Tracez chaque signal
for i, channel in enumerate(chans_to_plot, 1):
    # Extrayez le segment de signal pour l'intervalle spécifié
    segment_data = recording.get_traces(channel_ids=[str(channel)], start_frame=t_start_sample, end_frame=t_stop_sample)
    
    time_axis = np.linspace(30, 40, len(segment_data))
    
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

plt.tight_layout()
plt.show()


#%% Auto corr

import spikeinterface.widgets as sw
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

def select_directory():
    # Initialiser le GUI
    root = tk.Tk()
    
    # Cacher la fenêtre principale
    root.withdraw()
    
    # Ouvrir la fenêtre de sélection de dossier
    folder_selected = filedialog.askdirectory()
    
    # Fermer la GUI
    root.destroy()
    
    return folder_selected

# Utilisation
selected_folder = select_directory()
print(f"Folder selected: {selected_folder}")


for unit in unit_list:
    sw.plot_autocorrelograms(sorter_results, window_ms=150.0, bin_ms=1.0, unit_ids=np.array([unit]))
    plt.savefig(rf"{selected_folder}/Auutocorr_unit_{unit}.png")



#%% Cross corr

import spikeinterface.widgets as sw
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

def select_directory():
    # Initialiser le GUI
    root = tk.Tk()
    
    # Cacher la fenêtre principale
    root.withdraw()
    
    # Ouvrir la fenêtre de sélection de dossier
    folder_selected = filedialog.askdirectory()
    
    # Fermer la GUI
    root.destroy()
    
    return folder_selected

# Utilisation
selected_folder = select_directory()
print(f"Folder selected: {selected_folder}")


from itertools import combinations

# Supposons que unit_list contienne la liste de vos unités
num_units = len(unit_list)

# Divisez la liste d'unités en 4 groupes
group_size = num_units // 6
unit_groups = [unit_list[i:i + group_size] for i in range(0, num_units, group_size)]

# Générer toutes les combinaisons possibles 2 à 2 des groupes
combinations_2by2 = list(combinations(unit_groups, 2))

# Afficher toutes les combinaisons
for i,comb in enumerate(combinations_2by2):
    sw.plot_crosscorrelograms(sorter_results, unit_ids = np.concatenate(comb))
    
    plt.savefig(rf"{selected_folder}/Auutocorr_{i}.png")
    plt.close('all')
    


#%% plot spikes on trace
sw.plot_spikes_on_traces(we)
