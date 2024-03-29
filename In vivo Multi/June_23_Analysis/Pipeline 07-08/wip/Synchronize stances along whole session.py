# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 14:38:05 2023

@author: Gilles.DELBECQ
"""
import pandas as pd
import numpy as np
import os

from neo.core import SpikeTrain
from quantities import ms, s, Hz
from elephant.statistics import time_histogram, instantaneous_rate

from elephant import kernels
import matplotlib.pyplot as plt

# df_instantaneous_rate = pd.read_excel('D:/ePhy/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/instantaneous_rates.xlsx')

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
    import pickle
    
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
            if filename.lower().endswith(".xlsx") and session in filename and "Stance" in filename:
                csv_files.append(os.path.join(folderpath, filename))

    return csv_files


#%%Parameters
session_name = '0022_01_08'
mocap_session = "01"

spikesorting_results_path = r"\\equipe2-nas1\Public\DATA\Gilles\Spikesorting_August_2023\SI_Data\spikesorting_results"
concatenated_signals_path = r'\\equipe2-nas1\Public\DATA\Gilles\Spikesorting_August_2023\SI_Data\concatenated_signals'




sorter_name = "kilosort3"
sorter_folder = rf'{spikesorting_results_path}/{session_name}/{sorter_name}'
processing_data_path = rf"{sorter_folder}/curated\processing_data"


mocap_data_folder = r'\\equipe2-nas1\Public\DATA\Gilles\Spikesorting_August_2023\SI_Data\mocap_files\Auto-comp'

sampling_rate = 20000
mocap_freq = 200
mocap_delay = 45 #frames

# PArameters instantaneous rate
sampling_period = 50*ms
sampling_period_ir = 5*ms
sigma_ir =20*ms
kernel = kernels.GaussianKernel(sigma=sigma_ir, invert=True)


instantaneous_rate_bin_size = 1 #s
trageted_instantaneous_rate_bin_size = 0.005 #s

plot_check = False


#%%
# recordings_info = Get_recordings_info(session_name,concatenated_signals_path,spikesorting_results_path)

import pickle
recordings_info = pickle.load(open(rf"{concatenated_signals_path}/{session_name}/ttl_idx.pickle", "rb"))

"""
Load spike times
"""
# Load the spike times data
spike_times = pd.read_excel(rf"{processing_data_path}/spike_times.xlsx")

"""
Load Mocap data
"""
animal = session_name.split('_')[0]
print(rf"Loading MOCAP data for Mocap session {animal}_{mocap_session}")
mocap_files = list_recording_files(rf"{mocap_data_folder}\{animal}\Analysis",mocap_session)
# print(rf"{len(mocap_files)} trials found")

mocap_ttl = recordings_info['mocap_ttl_on'][::2]
# print(rf"{len(mocap_ttl)} TTL found in recordings info")

if len(mocap_ttl) > len(mocap_files):
    print(rf"Be careful ! there are more TTL ({len(mocap_ttl)}) than mocap files ({len(mocap_files)})")
elif len(mocap_ttl) < len(mocap_files):
    print(rf"Be careful ! there are less TTL ({len(mocap_ttl)}) than mocap files ({len(mocap_files)})")
    
mocap_ttl_times = mocap_ttl/sampling_rate


whole_data_mocap = []
for i,ttl_time in enumerate(mocap_ttl_times):
    mocap_file = None
    trial = i+1
    
    print(rf"Trial {trial}")
        
    for file_path in mocap_files:
        trial_file = int(file_path.split("_")[-1].split('.')[0])
        if trial_file == trial:
            mocap_file = file_path
         
    
    if mocap_file is not None:
        mocap_data = pd.read_excel(mocap_file).iloc[:, 1:]
        
        trial_start = round(ttl_time-mocap_delay/mocap_freq,3)
        trial_stop = round(trial_start + len(mocap_data) / mocap_freq,3)
        
        trial_time_axis = np.linspace(trial_start, trial_stop, len(mocap_data), endpoint=False)
        
        
        stance_left_times = (mocap_data['stance_left_idx']/mocap_freq)+trial_start
        stance_right_times = (mocap_data['stance_right_idx']/mocap_freq)+trial_start
        
        data_dict = {
            'stance_left_times' : stance_left_times,
            'stance_right_times' : stance_right_times
            }
        data_to_export = pd.DataFrame(data_dict)
                
        savepath = rf"{processing_data_path}\mocap_stances"
        Check_Save_Dir((savepath))
        
        data_to_export.to_excel(rf"{savepath}/{animal}_{mocap_session}_{trial}_stances.xlsx")