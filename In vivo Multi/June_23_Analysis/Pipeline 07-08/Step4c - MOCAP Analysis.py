# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 14:23:06 2023

@author: MOCAP
"""


import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import spikeinterface as si
import spikeinterface.extractors as se 
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.comparison as sc
import spikeinterface.exporters as sexp
import spikeinterface.widgets as sw
from spikeinterface.curation import MergeUnitsSorting, get_potential_auto_merge

from neo.core import SpikeTrain
from quantities import ms, s, Hz

from elephant.spike_train_generation import homogeneous_poisson_process, homogeneous_gamma_process
from elephant.statistics import mean_firing_rate
from elephant.statistics import time_histogram, instantaneous_rate
from elephant.kernels import GaussianKernel

import pickle
import time
import sys




#%%Parameters
session_name = '0026_02_08'
mocap_session = "01"

spikesorting_results_path = r"D:\ePhy\SI_Data\spikesorting_results"
concatenated_signals_path = r'D:\ePhy\SI_Data\concatenated_signals'
plots_path = r'D:\ePhy\SI_Data\plots'

sorter_name = "kilosort3"

sorter_folder = rf'{spikesorting_results_path}/{session_name}/{sorter_name}'
signal_folder = rf'{concatenated_signals_path}/{session_name}'


mocap_data_folder = 'D:\ePhy\SI_Data\mocap_files\Auto-comp'

sampling_rate = 20000*Hz
mocap_freq = 200
mocap_delay = 45 #frames

Save_plots = True
plot_inst_speed = True
plot_inst_feet = True
plot_dist_from_obstacle = True

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
    try:
        # Read the metadata file created during concatenation
        print("Reading the ttl_idx file in intan folder...")
        path = rf'{concatenated_signals_path}/{session_name}/'
        metadata = pickle.load(open(rf"{path}/ttl_idx.pickle", "rb"))
    except:
        print("No recording infos found in the intan folder. Please run Step 0")
    
    # save_path = rf'{spikesorting_results_path}/{session_name}/recordings_info.pickle'
    # if os.path.exists(save_path):
    #     print("Recordings info file exists")
    #     print("Loading info file...")
    #     recordings_info = pickle.load(open(save_path, "rb"))
    # else:
    #     print("Recordings info file does not exist")
    #     print("Getting info...")
    #     # Read the metadata file created during concatenation
    #     path = rf'{concatenated_signals_path}/{session_name}/'
    #     metadata = pickle.load(open(rf"{path}/ttl_idx.pickle", "rb"))
       
    #     # Loop over intan files
    #     recordings_list = metadata['recordings_files']
    #     # RHD file reading
    #     multi_recordings, recordings_lengths, multi_stim_idx, multi_frame_idx, frame_start_delay = [], [], [], [], []
        
    #     # Concatenate recordings
    #     for record in recordings_list:
    #         reader = read_data(record)
    #         signal = reader['amplifier_data']
    #         recordings_lengths.append(len(signal[0]))
    #         multi_recordings.append(signal)
            
    #         stim_idx = reader['board_dig_in_data'][0]  # Digital data for stim of the file
    #         multi_stim_idx.append(stim_idx)  # Digital data for stim of all the files
            
    #         frame_idx = reader['board_dig_in_data'][1]  # Get digital data for mocap ttl
    #         multi_frame_idx.append(frame_idx)  # Digital data for mocap ttl of all the files
            
    #     anaglog_signal_concatenated = np.hstack(multi_recordings)  # Signal concatenated from all the files
    #     digital_stim_signal_concatenated = np.hstack(multi_stim_idx)  # Digital data for stim concatenated from all the files
    #     digital_mocap_signal_concatenated = np.hstack(multi_frame_idx)
        
    #     # Get sampling freq
    #     sampling_rate = reader['frequency_parameters']['amplifier_sample_rate']
        
    #     recordings_lengths_cumsum = np.cumsum(np.array(recordings_lengths) / sampling_rate)
                                              
    #     # Return: recording length, recording length cumsum, digital signals 1 and 2 (in full or logical?)
    #     # Save them in a pickle
        
    #     recordings_info = {
    #         'recordings_length': recordings_lengths,
    #         'recordings_length_cumsum': recordings_lengths_cumsum,
    #         'sampling_rate': sampling_rate,
    #         'digital_stim_signal_concatenated': digital_stim_signal_concatenated,
    #         'digital_mocap_signal_concatenated': digital_mocap_signal_concatenated
    #     }
        
    #     pickle.dump(recordings_info, open(save_path, "wb"))
        
    print('Done')
    return metadata

# def list_recording_files(path):
#     """
#     List all recording files (.rhd) in the specified directory and its subdirectories.
    
#     Parameters:
#         path (str): The directory path to search for recording files.
        
#     Returns:
#         list: A list of paths to all recording files found.
#     """
#     import os
#     import glob
#     subfolders = [ f.path for f in os.scandir(path) if f.is_file() ]
    
#     return subfolders

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
    # Créer une fenêtre de moyenne mobile
    window = np.ones(window_size) / window_size

    # Appliquer la moyenne mobile en utilisant la convolution
    smoothed_data = np.convolve(data, window, mode='same')

    return smoothed_data


#%% Loadings
#Load units
recordings_info = Get_recordings_info(session_name,concatenated_signals_path,spikesorting_results_path)

print(rf"Loading spikesorting results for session {session_name}")
sorter_results = ss.NpzSortingExtractor.load_from_folder(rf'{sorter_folder}/curated').remove_empty_units()
signal = si.load_extractor(signal_folder)
we = si.WaveformExtractor(signal,sorter_results)

time_axis = signal.get_times()
unit_list = sorter_results.get_unit_ids()

print(rf"{len(sorter_results.get_unit_ids())} units loaded")


#Load Mocap data
animal = session_name.split('_')[0]
print(rf"Loading MOCAP data for Mocap session {animal}_{mocap_session}")
mocap_files = list_recording_files(rf"{mocap_data_folder}/{animal}/Analysis",mocap_session)
# print(rf"{len(mocap_files)} trials found")

mocap_ttl = recordings_info['mocap_ttl_on'][::2]
# print(rf"{len(mocap_ttl)} TTL found in recordings info")

if len(mocap_ttl) > len(mocap_files):
    print(rf"Be careful ! there are more TTL ({len(mocap_ttl)}) than mocap files ({len(mocap_files)})")
elif len(mocap_ttl) < len(mocap_files):
    print(rf"Be careful ! there are less TTL ({len(mocap_ttl)}) than mocap files ({len(mocap_files)})")
    
mocap_ttl_times = mocap_ttl/sampling_rate
    

#%% Splitting
#Split spiketrains by Mocap TTL
for i,ttl_time in enumerate(mocap_ttl_times):
    mocap_file = None
    trial = i+1
    print(rf"############################## Trial {trial} ########################################")
    
    for file_path in mocap_files:
        trial_file = int(file_path.split("_")[-1].split('.')[0])
        if trial_file == trial:
            mocap_file = file_path
            
    if mocap_file == None:
        print(rf"No mocap file corresponding to the trial {trial}")
    else:
        mocap_data = pd.read_excel(mocap_file)
        time_length = len(mocap_data)/mocap_freq
        
        
        
        
        for unit in unit_list:
            try:
                
                spike_times = sorter_results.get_unit_spike_train(unit_id=unit)/sampling_rate*Hz*s
                selected_spike_times = spike_times[(spike_times >= ttl_time*Hz*s) & (spike_times <= ttl_time*Hz*s+time_length*s)]
                
                spiketrain = SpikeTrain(selected_spike_times,t_start = ttl_time*Hz*s, t_stop=ttl_time*Hz*s+time_length*s)     
                inst_rate = instantaneous_rate(spiketrain, sampling_period=10*ms)
                
                speed = mocap_data['speed_back1']
                speed_right_foot =  mocap_data['speed_right_foot']
                speed_left_foot =  mocap_data['speed_left_foot']
                
                distance_from_obstacle = mocap_data['distance_from_obstacle']
                
                mocap_time_axis = (np.array(range(len(mocap_data)))/200+ttl_time*Hz)-mocap_delay/mocap_freq
                
                histogram_count = time_histogram([spiketrain], 0.05*s)
                histogram_rate = time_histogram([spiketrain],  0.05*s, output='rate')
                
                
                
                
                
                
                # plt.figure() 
                # plt.bar(histogram_rate.times, histogram_rate.magnitude.flatten(), width=histogram_rate.sampling_period,
                #         align='edge', alpha=0.3, label='time histogram (rate)',color='black')
            
                
                
                # # Instantaneous rate
                # plt.plot(inst_rate.times.rescale(s), inst_rate.rescale(histogram_rate.dimensionality).magnitude.flatten(), label='instantaneous rate')
                
                
                # # axis labels and legend
                # plt.xlabel('time [{}]'.format(spiketrain.times.dimensionality.latex))
                # plt.ylabel('firing rate [{}]'.format(histogram_rate.dimensionality.latex))
                
                # plt.xlim(spiketrain.t_start, spiketrain.t_stop)   
                
                # #plot speed
                # plt.plot(speed)
                
                # plt.legend()
                # plt.title(rf'Unit #{unit}')
                # plt.show()
                
                 
                if plot_inst_speed == True:
                    
                    # Créer une figure et un axe principal
                    fig, ax1 = plt.subplots()
                    
                    # Tracé de l'histogramme de taux sur l'axe principal
                    ax1.bar(histogram_rate.times, histogram_rate.magnitude.flatten(), width=histogram_rate.sampling_period,
                            align='edge', alpha=0.3, label='time histogram (rate)',color='black')
                    ax1.plot(inst_rate.times.rescale(s), inst_rate.rescale(histogram_rate.dimensionality).magnitude.flatten(), label='instantaneous rate')
                    
                    # Configurer les étiquettes, les titres et les légendes pour l'axe principal
                    ax1.set_xlabel('Temps [s]')
                    ax1.set_ylabel('Taux de décharge [Hz]')
                    ax1.set_title(rf'Trial # {trial} - Unit # {unit} ')
                    ax1.legend(loc='upper left')
                    
                    # Créer un axe Y secondaire pour la vitesse
                    ax2 = ax1.twinx()
                    
                    # Tracé de la vitesse sur l'axe Y secondaire
                    ax2.plot(mocap_time_axis, moving_average(np.array(speed),10), color='red', label='Vitesse')
                    
                    # Configurer les étiquettes et la légende pour l'axe Y secondaire
                    ax2.set_ylabel('Vitesse [m/s]', color='red')
                    ax2.tick_params(axis='y', labelcolor='red')
                    ax2.legend(loc='upper right')
                    
                    # ax1.set_xlim(mocap_time_axis[np.argmax(~np.isnan(speed))], mocap_time_axis[-1])
                    
                    index_first_non_nan = next((index for index, value in enumerate(speed) if not np.isnan(value)), None)
                    index_last_non_nan = len(speed) - 1 - next((index for index, value in enumerate(speed[::-1]) if not np.isnan(value)), None)
                    ax1.set_xlim(mocap_time_axis[index_first_non_nan], mocap_time_axis[index_last_non_nan])
                    
                    # Afficher le tracé
                    # plt.show()
                    
                    
                    if Save_plots == True:
                        
                        savefig_path = rf'{plots_path}/{animal}/Session_{mocap_session}/Speed/Inst_rate_mocap_{animal}_{mocap_session}_{trial}_Unit_{unit}.png'
                        Check_Save_Dir(os.path.dirname(savefig_path))
                        plt.savefig(savefig_path)
                    
                    plt.close()
                    
                    # correlation = np.correlate(inst_rate.magnitude.flatten(),np.array(speed),mode='valid')
    
    
                    # Supposons que 'array1' et 'array2' sont vos tableaux 1D avec des valeurs, y compris NaN
                    
                    # Traiter les tableaux pour masquer les valeurs NaN
                    masked_array1 = np.ma.masked_invalid(inst_rate.magnitude.flatten())
                    masked_array2 = np.ma.masked_invalid(np.array(speed))
                    
                    # Calculer la corrélation croisée en ignorant les valeurs NaN
                    correlation = np.correlate(masked_array2, masked_array1, mode='valid')
                                   
                    
                    
                    
                    
                    # plt.figure()
                    # plt.title(rf'Correlation Trial # {trial} - Unit # {unit} ')
                    # plt.plot(correlation)
                    # if Save_plots == True:
                    #     savefig_path = rf'{plots_path}/{animal}/Session_{mocap_session}/Speed/Correlation_{animal}_{mocap_session}_{trial}_Unit_{unit}.png'
                    #     Check_Save_Dir(os.path.dirname(savefig_path))
                    #     plt.savefig(savefig_path)
                    # plt.close()
                                    
                if plot_inst_feet == True:                   
                    # Créer une figure et un axe principal
                    fig, ax1 = plt.subplots()
                    
                    # Tracé de l'histogramme de taux sur l'axe principal
                    ax1.bar(histogram_rate.times, histogram_rate.magnitude.flatten(), width=histogram_rate.sampling_period,
                            align='edge', alpha=0.3, label='time histogram (rate)',color='black')
                    ax1.plot(inst_rate.times.rescale(s), inst_rate.rescale(histogram_rate.dimensionality).magnitude.flatten(), label='instantaneous rate')
                    
                    # Configurer les étiquettes, les titres et les légendes pour l'axe principal
                    ax1.set_xlabel('Temps [s]')
                    ax1.set_ylabel('Taux de décharge [Hz]')
                    ax1.set_title(rf'Trial # {trial} - Unit # {unit} ')
                    ax1.legend(loc='upper left')
                    
                    # Créer un axe Y secondaire pour la vitesse
                    ax2 = ax1.twinx()
                    
                    # Tracé de la vitesse sur l'axe Y secondaire
                    ax2.plot(mocap_time_axis, moving_average(np.array(speed_right_foot),10), alpha=0.5, color='blue', label='Vitesse')
                    ax2.plot(mocap_time_axis, moving_average(np.array(speed_left_foot),10),alpha=0.5, color='red', label='Vitesse')
                    
                    # Configurer les étiquettes et la légende pour l'axe Y secondaire
                    ax2.set_ylabel('Vitesse [m/s]', color='red')
                    ax2.tick_params(axis='y', labelcolor='red')
                    ax2.legend(loc='upper right')
                    
                    ax1.set_xlim(mocap_time_axis[np.argmax(~np.isnan(speed_right_foot))], mocap_time_axis[-1])
                    
                    # Afficher le tracé
                    # plt.show()
                    
                    
                    if Save_plots == True:
                        
                        savefig_path = rf'{plots_path}/{animal}/Session_{mocap_session}/Foot_Speed/Foot_Speed_rate_mocap_{animal}_{mocap_session}_{trial}_Unit_{unit}.png'
                        Check_Save_Dir(os.path.dirname(savefig_path))
                        plt.savefig(savefig_path)
                    
                    plt.close()
                    
                    # correlation = np.correlate(inst_rate.magnitude.flatten(),np.array(speed),mode='valid')
    
    
                    # Supposons que 'array1' et 'array2' sont vos tableaux 1D avec des valeurs, y compris NaN
                    
                    # Traiter les tableaux pour masquer les valeurs NaN
                    # masked_array1 = np.ma.masked_invalid(inst_rate.magnitude.flatten())
                    # masked_array2 = np.ma.masked_invalid(np.array(speed))
                    
                    # # Calculer la corrélation croisée en ignorant les valeurs NaN
                    # correlation = np.correlate(masked_array2, masked_array1, mode='valid')
                                   
                    
                    
                    
                    
                    # plt.figure()
                    # plt.title(rf'Correlation Trial # {trial} - Unit # {unit} ')
                    # plt.plot(correlation)
                    # if Save_plots == True:
                    #     savefig_path = rf'{plots_path}/{animal}/{mocap_session}/Correlation_{animal}_{mocap_session}_{trial}_Unit_{unit}.png'
                    #     Check_Save_Dir(os.path.dirname(savefig_path))
                    #     plt.savefig(savefig_path)
                    # plt.close()
                
                if plot_dist_from_obstacle == True:                   
                    # Créer une figure et un axe principal
                    fig, ax1 = plt.subplots()
                    
                    # Tracé de l'histogramme de taux sur l'axe principal
                    ax1.bar(histogram_rate.times, histogram_rate.magnitude.flatten(), width=histogram_rate.sampling_period,
                            align='edge', alpha=0.3, label='time histogram (rate)',color='black')
                    ax1.plot(inst_rate.times.rescale(s), inst_rate.rescale(histogram_rate.dimensionality).magnitude.flatten(), label='instantaneous rate')
                    
                    # Configurer les étiquettes, les titres et les légendes pour l'axe principal
                    ax1.set_xlabel('Temps [s]')
                    ax1.set_ylabel('Taux de décharge [Hz]')
                    ax1.set_title(rf'Trial # {trial} - Unit # {unit} ')
                    ax1.legend(loc='upper left')
                    
                    # Créer un axe Y secondaire pour la vitesse
                    ax2 = ax1.twinx()
                    
                    # Tracé de la vitesse sur l'axe Y secondaire
                    ax2.plot(mocap_time_axis, distance_from_obstacle, alpha=0.5, color='blue', label='Vitesse')

                    
                    # Configurer les étiquettes et la légende pour l'axe Y secondaire
                    ax2.set_ylabel('Vitesse [m/s]', color='red')
                    ax2.tick_params(axis='y', labelcolor='red')
                    ax2.legend(loc='upper right')
                    
                    ax1.set_xlim(mocap_time_axis[np.argmax(~np.isnan(speed_right_foot))], mocap_time_axis[-1])
                    
                    # Afficher le tracé
                    # plt.show()
                    
                    
                    if Save_plots == True:
                        
                        savefig_path = rf'{plots_path}/{animal}/Session_{mocap_session}/Distance_from_obstacle/Dist_from_obst_rate_mocap_{animal}_{mocap_session}_{trial}_Unit_{unit}.png'
                        Check_Save_Dir(os.path.dirname(savefig_path))
                        plt.savefig(savefig_path)
                    
                    plt.close()
                
            except:
                print(rf"Error with unit # {unit}")
        
    




#Split Mocap Data by Mocap TTL

