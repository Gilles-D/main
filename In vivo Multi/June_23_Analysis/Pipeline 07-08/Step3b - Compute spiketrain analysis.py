# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:19:06 2023

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
plots_path = r'D:\ePhy\SI_Data\plots'

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


#%%Compute instantaneous rate

inst_rates = []

sampling_period = 5*ms
kernel = kernels.AlphaKernel(sigma=1*s, invert=True)

whole_spike_times = np.array()
for unit in unit_list:
    spike_times = (sorter_results.get_unit_spike_train(unit_id=unit)/sampling_rate)*s
    whole_spike_times.append(spike_times)
    
max_length = max(len(arr) for arr in whole_spike_times)
data = {f'Unit_{unit_list[i]}': np.pad(arr, (0, max_length - len(arr)), mode='constant', constant_values=np.nan) for i, arr in enumerate(whole_spike_times)}

df_data = pd.DataFrame(data)
df_data.to_excel(rf"{sorter_folder}\curated\processing_data\spike_times.xlsx")


for unit in unit_list:
    print(rf"Unit {unit}")
    spike_times = (sorter_results.get_unit_spike_train(unit_id=unit)/sampling_rate)*s
    total_duration = signal.get_total_duration()*s
    time_axis = np.array(range(0,int(total_duration*20000)))/20000
    
    spiketrain = SpikeTrain(spike_times, t_stop=total_duration)

    inst_rate = instantaneous_rate(spiketrain, sampling_period,kernel=kernel)
    inst_rates.append(inst_rate)
    
    
    
    # plt.plot(time_axis_instantaneous_rate,inst_rate)
    

time_axis_instantaneous_rate = range(len(inst_rates[0]))/(1/sampling_period)/1000
inst_rates_array = np.hstack(inst_rates)

df_inst_rates = pd.DataFrame(inst_rates_array)
df_inst_rates.index = time_axis_instantaneous_rate
df_inst_rates.columns = unit_list


savepath = rf"{sorter_folder}\curated\processing_data\instantaneous_rates.xlsx"
Check_Save_Dir(os.path.dirname((savepath)))

df_inst_rates.to_excel(savepath)

