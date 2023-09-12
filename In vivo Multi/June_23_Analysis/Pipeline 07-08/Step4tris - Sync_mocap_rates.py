# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:55:51 2023

@author: MOCAP
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
            if filename.lower().endswith(".xlsx") and session in filename and "Analysis" in filename:
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

# instantaneous_rates_df = pd.read_excel(rf"{processing_data_path}/instantaneous_rates_20ms.xlsx")
# unit_list = instantaneous_rates_df.columns.tolist()[1:]
# instantaneous_rates_time_taxis = np.array(instantaneous_rates_df[instantaneous_rates_df.columns[0]])

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

        
        # trial_time_axis = np.arange(trial_start, trial_stop, 1/mocap_freq)
        
        mocap_data.insert(0,'time_axis',trial_time_axis)
        
        
        
        instantaneous_rates = []
        unit_list = spike_times.columns.tolist()[1:]
        
        selected_spikes = spike_times.apply(lambda col: col[(col >= trial_start) & (col <= trial_stop)], axis=0)
        selected_spikes = selected_spikes.drop(columns=["Unnamed: 0"])
        
        
        for unit in selected_spikes.columns:
            # print(unit)
            selected_unit_spikes = selected_spikes[unit]
            spiketrain = SpikeTrain(np.array(selected_unit_spikes)*s,t_start = trial_start*s, t_stop=trial_stop*s)

            inst_rate = instantaneous_rate(spiketrain, sampling_period_ir,kernel=kernel)
            
            
            if plot_check == True:
                histogram_count = time_histogram([spiketrain], sampling_period)
                histogram_rate = time_histogram([spiketrain], sampling_period, output='rate')
                
                plt.figure(dpi=150)
    
                # time histogram
                plt.bar(histogram_rate.times.rescale(ms), histogram_rate.magnitude.flatten(), width=histogram_rate.sampling_period, align='edge', alpha=0.3, label='time histogram (rate)')
                
                # instantaneous rate
                plt.plot(inst_rate.times.rescale(ms), inst_rate.rescale(histogram_rate.dimensionality).magnitude.flatten(), label='instantaneous rate')
                
                plt.title(rf"{unit}")
                
            instantaneous_rates.append(inst_rate.magnitude.flatten())
            
        
            
        
        instantaneous_rates_array = np.array(instantaneous_rates).T
        instantaneous_rates_df = pd.DataFrame(instantaneous_rates_array,columns=unit_list)       
        
        if len(mocap_data) == len(inst_rate):
            instantaneous_rates_df['time_axis'] =  mocap_data['time_axis']
            
        
        elif len(mocap_data) != len(inst_rate):
            print("Different time axis size")
            print(rf"mocap length = {len(mocap_data)} rate length = {len(inst_rate)}")
            
            instantaneous_rates_df['time_axis'] = trial_time_axis[0:-1]
            mocap_data = mocap_data[:-1]
            
                       
            
        instantaneous_rates_df = instantaneous_rates_df.set_index('time_axis')
        mocap_data = mocap_data.set_index('time_axis')     
        
        savepath = rf"{processing_data_path}\sync_data_rate_sigma_{sigma_ir}ms_Gaussian"
        Check_Save_Dir((savepath))
        
        mocap_data.to_excel(rf"{savepath}/{animal}_{mocap_session}_{trial}_mocap.xlsx")
        instantaneous_rates_df.to_excel(rf"{savepath}/{animal}_{mocap_session}_{trial}_rates.xlsx")