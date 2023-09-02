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
from elephant import kernels

import pickle
import time
import sys

import seaborn as sns
import scipy.cluster.hierarchy as sch


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

Save_plots = False
plot_inst_speed = False
plot_inst_feet = False
plot_dist_from_obstacle = False
plot_back_inclination = False

do_correlations = True

concatenate_by_unit = True

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
trial_list = []
correlation_matrix_speed, correlation_matrix_obst, correlation_matrix_z = [],[],[]

concatenated_spiketrains_by_unit =[]

for i,ttl_time in enumerate(mocap_ttl_times):
    mocap_file = None
    trial = i+1
    print(rf"############################## Trial {trial} ########################################")
    correlation_trial_speed,correlation_trial_obstacle,correlation_trial_Z=[],[],[]
    
     
       
    for file_path in mocap_files:
        trial_file = int(file_path.split("_")[-1].split('.')[0])
        if trial_file == trial:
            mocap_file = file_path
            
    if mocap_file == None:
        print(rf"No mocap file corresponding to the trial {trial}")
    else:
        mocap_data = pd.read_excel(mocap_file)
        time_length = len(mocap_data)/mocap_freq
        trial_list.append(trial)
        
        for unit in unit_list:
            try:
                """
                Computing :
                 - spike train
                 - instantaneous rate
                 - speeds
                 - histogram
                
                """
                spike_times = sorter_results.get_unit_spike_train(unit_id=unit)/sampling_rate*Hz*s
                selected_spike_times = spike_times[(spike_times >= ttl_time*Hz*s-(mocap_delay/mocap_freq)*s) & (spike_times <= ttl_time*Hz*s+time_length*s-(mocap_delay/mocap_freq)*s)]
                
                spiketrain = SpikeTrain(selected_spike_times,t_start = ttl_time*Hz*s-(mocap_delay/mocap_freq)*s, t_stop=ttl_time*Hz*s+time_length*s-(mocap_delay/mocap_freq)*s)   
                
                inst_rate = instantaneous_rate(spiketrain, sampling_period=5*ms)
                
                kernel = kernels.AlphaKernel(sigma=0.05*s, invert=True)
                sampling_period = 5*ms
                inst_rate2 = instantaneous_rate(spiketrain, sampling_period,kernel=kernel)
                
                speed = mocap_data['speed_back1']
                acceleration =  np.abs(np.diff(speed) / 1*s)
                
                speed_right_foot =  mocap_data['speed_right_foot']
                speed_left_foot =  mocap_data['speed_left_foot']
                
                distance_from_obstacle = mocap_data['distance_from_obstacle']
                
                
                
                mocap_time_axis = (np.array(range(len(mocap_data)))/200+ttl_time*Hz)-mocap_delay/mocap_freq
                
                histogram_count = time_histogram([spiketrain], 0.05*s)
                histogram_rate = time_histogram([spiketrain],  0.05*s, output='rate')
                
                    
                
                
                """
                Correlations computation
                """
                
                if do_correlations == True:
                    """
                    Compute correlation between inst_rate and speed
                    """
                    #Mocap delay should be fixed
                    
                    index_first_non_nan = next((index for index, value in enumerate(speed) if not np.isnan(value)), None)
                    index_last_non_nan = len(speed) - 1 - next((index for index, value in enumerate(speed[::-1]) if not np.isnan(value)), None)
                    selected_speed = np.array(speed[index_first_non_nan:index_last_non_nan])
                    
                    non_nan_indices = np.where(~np.isnan(selected_speed))[0]
                    nan_indices = np.where(np.isnan(selected_speed))[0]
                    
                    interpolated_values = np.interp(nan_indices, non_nan_indices, selected_speed[non_nan_indices])
                    array_with_interpolated_values = selected_speed.copy()
                    array_with_interpolated_values[nan_indices] = interpolated_values
                    
                    
                    # selected_inst_rate = inst_rate.magnitude.flatten()[index_first_non_nan:index_last_non_nan]
                    selected_inst_rate = inst_rate2.magnitude.flatten()[index_first_non_nan:index_last_non_nan]
                    corr = np.correlate(array_with_interpolated_values,selected_inst_rate)
                    corr_coef = np.corrcoef(selected_inst_rate,array_with_interpolated_values)
                    
                    correlation_trial_speed.append(corr_coef[0,1])
                    
                    
                    """
                    Compute correlation between inst_rate and distance from obstacle
                    """
                    if np.isnan(distance_from_obstacle).all() == False:
                        index_first_non_nan = next((index for index, value in enumerate(distance_from_obstacle) if not np.isnan(value)), None)
                        index_last_non_nan = len(distance_from_obstacle) - 1 - next((index for index, value in enumerate(distance_from_obstacle[::-1]) if not np.isnan(value)), None)
                        selected_speed = np.array(distance_from_obstacle[index_first_non_nan:index_last_non_nan])
                        
                        non_nan_indices = np.where(~np.isnan(selected_speed))[0]
                        nan_indices = np.where(np.isnan(selected_speed))[0]
                        
                        interpolated_values = np.interp(nan_indices, non_nan_indices, selected_speed[non_nan_indices])
                        array_with_interpolated_values = selected_speed.copy()
                        array_with_interpolated_values[nan_indices] = interpolated_values
                        
                        
                        selected_inst_rate = inst_rate2.magnitude.flatten()[index_first_non_nan:index_last_non_nan]
                        corr = np.correlate(array_with_interpolated_values,selected_inst_rate)
                        corr_coef = np.corrcoef(selected_inst_rate,array_with_interpolated_values)
                        
                        correlation_trial_obstacle.append(corr_coef[0,1])
                        
                    
                    else:
                        correlation_trial_obstacle.append(np.nan)
                        
                        
                    """
                    Compute correlation between inst_rate and dback1_Z
                    """
                    
                    back1_z = mocap_data['back_1_Z']
                    
                    index_first_non_nan = next((index for index, value in enumerate(back1_z) if not np.isnan(value)), None)
                    index_last_non_nan = len(back1_z) - 1 - next((index for index, value in enumerate(back1_z[::-1]) if not np.isnan(value)), None)
                    selected_speed = np.array(back1_z[index_first_non_nan:index_last_non_nan])
                    
                    non_nan_indices = np.where(~np.isnan(selected_speed))[0]
                    nan_indices = np.where(np.isnan(selected_speed))[0]
                    
                    interpolated_values = np.interp(nan_indices, non_nan_indices, selected_speed[non_nan_indices])
                    array_with_interpolated_values = selected_speed.copy()
                    array_with_interpolated_values[nan_indices] = interpolated_values
                    
                    
                    selected_inst_rate = inst_rate2.magnitude.flatten()[index_first_non_nan:index_last_non_nan]
                    corr = np.correlate(array_with_interpolated_values,selected_inst_rate)
                    corr_coef = np.corrcoef(selected_inst_rate,array_with_interpolated_values)
                    
                    correlation_trial_Z.append(corr_coef[0,1])
                    
               
                
                """
                Plot the speed with inst_rate
                """
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
                    ax2.plot(mocap_time_axis, moving_average(np.array(speed),10),alpha=0.5, color='red', label='Vitesse')
                    # ax2.plot(mocap_time_axis[1:], moving_average(np.array(acceleration),10),alpha=0.5, color='green', label='Vitesse')
                    
                    # Configurer les étiquettes et la légende pour l'axe Y secondaire
                    ax2.set_ylabel('Vitesse [m/s]', color='red')
                    ax2.tick_params(axis='y', labelcolor='red')
                    ax2.legend(loc='upper right')
                    
                    # ax1.set_xlim(mocap_time_axis[np.argmax(~np.isnan(speed))], mocap_time_axis[-1])
                    
                    index_first_non_nan = next((index for index, value in enumerate(speed) if not np.isnan(value)), None)
                    index_last_non_nan = len(speed) - 1 - next((index for index, value in enumerate(speed[::-1]) if not np.isnan(value)), None)
                    ax1.set_xlim(mocap_time_axis[index_first_non_nan]-2, mocap_time_axis[index_last_non_nan]+2)
                    
                    # Afficher le tracé
                    # plt.show()
                    
                    
                    if Save_plots == True:
                        
                        savefig_path = rf'{plots_path}/{animal}/Session_{mocap_session}/Speed/Inst_rate_mocap_{animal}_{mocap_session}_{trial}_Unit_{unit}.png'
                        Check_Save_Dir(os.path.dirname(savefig_path))
                        plt.savefig(savefig_path)
                    
                    plt.close()
                    
                    
                """
                Plot speed of feet with inst rate
                
                """                   
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
                    
                
                """
                Plot distance from obstacle
                """
                
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
                    plt.show()
                    
                    
                    if Save_plots == True:
                        
                        savefig_path = rf'{plots_path}/{animal}/Session_{mocap_session}/Distance_from_obstacle/Dist_from_obst_rate_mocap_{animal}_{mocap_session}_{trial}_Unit_{unit}.png'
                        Check_Save_Dir(os.path.dirname(savefig_path))
                        plt.savefig(savefig_path)
                    
                    plt.close()
                
                """
                Plot Back inclination
                
                """
                if plot_back_inclination == True:
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
                    
                    back_inclination = mocap_data['back_inclination']
                    
                    ax2.plot(mocap_time_axis, back_inclination, alpha=0.5, color='blue', label='z inclination')
                    ax2.plot(mocap_time_axis, mocap_data['back_1_Z'], alpha=0.5, color='green', label='back_1_Z')
                    ax2.plot(mocap_time_axis, mocap_data['back_2_Z'], alpha=0.5, color='red', label='back_2_Z')
                    
                    # Configurer les étiquettes et la légende pour l'axe Y secondaire
                    ax2.set_ylabel('Vitesse [m/s]', color='red')
                    ax2.tick_params(axis='y', labelcolor='red')
                    ax2.legend(loc='upper right')
                    
                    ax1.set_xlim(mocap_time_axis[np.argmax(~np.isnan(speed_right_foot))], mocap_time_axis[-1])
                    
                    # Afficher le tracé
                    # plt.show()
                    
                    
                    if Save_plots == True:
                        
                        savefig_path = rf'{plots_path}/{animal}/Session_{mocap_session}/Back_inclination/Back_inclination_rate_mocap_{animal}_{mocap_session}_{trial}_Unit_{unit}.png'
                        Check_Save_Dir(os.path.dirname(savefig_path))
                        plt.savefig(savefig_path)
                    
                    plt.close()
                    
            except:
                print(rf"Error with unit # {unit}")
                if do_correlations == True:
                    correlation_trial_speed.append(np.nan)
                    correlation_trial_obstacle.append(np.nan)
                    correlation_trial_Z.append(np.nan)
            
            plt.close('all')

        if do_correlations == True:
            correlation_matrix_speed.append(correlation_trial_speed)
            correlation_matrix_obst.append(correlation_trial_obstacle)
            correlation_matrix_z.append(correlation_trial_Z)
        
        
        
        
if do_correlations == True:
    correlation_save_path = rf'{sorter_folder}/curated/correlation.xlsx'

    correlation_df_speed = pd.DataFrame(np.array(correlation_matrix_speed),index=trial_list, columns=unit_list)
    correlation_df_obst = pd.DataFrame(np.array(correlation_matrix_obst),index=trial_list, columns=unit_list)
    correlation_df_z = pd.DataFrame(np.array(correlation_matrix_z),index=trial_list, columns=unit_list)

    with pd.ExcelWriter(correlation_save_path) as writer:
        correlation_df_speed.to_excel(writer, sheet_name="Speed", index=False)
        correlation_df_obst.to_excel(writer, sheet_name="Obstacle", index=False)
        correlation_df_z.to_excel(writer, sheet_name="Back_Z", index=False)

#%% Correlation plots

#TODO : add possibility to load correlation array if already exists without running previosu session


correlation_df = correlation_df_speed
correlation_matrix = correlation_matrix_speed

column_means = correlation_df.mean()
column_med = correlation_df.median()

"""
Plot units by mean of correlation over trials
"""

plt.figure()
plt.title("Correlation between inst_rate and speed (mean)")
plt.xlabel("Units #")
plt.ylabel("Corr coeff")

sns.boxplot(data = correlation_df[column_means.sort_values().index])    
plt.savefig(rf'{sorter_folder}/curated/correlation_mean.png')

"""
Plot units by median of correlation over trials
"""

plt.figure()
plt.title("Correlation between inst_rate and speed (median)")
plt.xlabel("Units #")
plt.ylabel("Corr coeff")

sns.boxplot(data = correlation_df[column_med.sort_values().index])  
plt.savefig(rf'{sorter_folder}/curated/correlation_med.png')

"""
Plot dendrogramm by hierarchichal clustering
"""

plt.figure()
plt.title(rf"Dendogramme {session_name}")

correlation_matrix_masked = np.ma.masked_array(np.array(correlation_matrix), mask=np.isnan(correlation_matrix))
array_without_nan = np.nan_to_num(np.array(correlation_matrix), nan=0)

distance_matrix = np.sqrt(2 * (1 - np.array(array_without_nan).T))
# Effectuer le clustering hiérarchique avec des noms d'unités en abscisses

dendrogram = sch.dendrogram(sch.linkage(distance_matrix, method='ward'),labels=unit_list)
plt.xlabel("Units #")
# Afficher le dendrogramme
plt.savefig(rf'{sorter_folder}/curated/dendrogram.png')

plt.show()


# from sklearn.cluster import KMeans

# # Transposer la matrice de corrélation
# correlation_matrix_transposed = correlation_array.T

# # Effectuer le regroupement K-means
# k = 3  # Nombre de classes souhaitées
# kmeans = KMeans(n_clusters=k)
# cluster_labels = kmeans.fit_predict(correlation_matrix_transposed)

# # Créer un DataFrame pour stocker les résultats du clustering
# clustering_results = pd.DataFrame({'Unit': unit_list, 'Cluster': cluster_labels})

#%% Concatenante all session by unit
# Loop on units
# Select spike train whole session

# Get mocap infos from all trials of session
# Get time axis of whole session
# Append all mocap trials, with nan? bewteen trials

for unit in unit_list:
    print(rf"Unit {unit}")
    spike_times = sorter_results.get_unit_spike_train(unit_id=unit)/sampling_rate*Hz*s
    total_duration = signal.get_total_duration()
    time_axis = np.array(range(0,int(total_duration*20000)))/20000
    
    spiketrain = SpikeTrain(spike_times, t_stop=total_duration)
    
    
    
    sampling_period = 2*ms

    kernel = kernels.AlphaKernel(sigma=0.05*s, invert=True)
    inst_rate = instantaneous_rate(spiketrain, sampling_period,kernel=kernel)  

    
    
    for i,ttl_time in enumerate(mocap_ttl_times):
        mocap_file = None
        trial = i+1
        
        print(rf"Trial {trial}")
            
        for file_path in mocap_files:
            trial_file = int(file_path.split("_")[-1].split('.')[0])
            if trial_file == trial:
                mocap_file = file_path
             
        
        if mocap_file is not None:
            speed = pd.read_excel(mocap_file)['speed_back1']
            trial_time_axis = (np.array(range(len(speed)))/mocap_freq*s)+ttl_time-mocap_delay/mocap_freq*s
            
            selected_spike_times = spike_times[(spike_times >= ttl_time*Hz*s) & (spike_times <= ttl_time*Hz*s+time_length*s)]
            
            
            
            data_matrix = np.column_stack((trial_time_axis,speed))
            try:
                whole_mocap_data = np.vstack((whole_mocap_data,data_matrix))
            except NameError:
                whole_mocap_data = data_matrix
            del data_matrix
        
        else:
            print(rf"Pas de fichier pour l'essai {trial}")
            
    

   
    # Créer une figure et un axe principal
    fig, ax1 = plt.subplots()
    
    # Tracé de l'histogramme de taux sur l'axe principal
    ax1.plot(inst_rate.times.rescale(s), inst_rate.rescale(histogram_rate.dimensionality).magnitude.flatten(), label='instantaneous rate')
    
    # Configurer les étiquettes, les titres et les légendes pour l'axe principal
    ax1.set_xlabel('Temps [s]')
    ax1.set_ylabel('Taux de décharge [Hz]')
    ax1.set_title(rf'Unit # {unit} ')
    ax1.legend(loc='upper left')
    
    # Créer un axe Y secondaire pour la vitesse
    ax2 = ax1.twinx()
    
    # Tracé de la vitesse sur l'axe Y secondaire
    ax2.plot(whole_mocap_data[:,0], moving_average(whole_mocap_data[:,1],10),alpha=0.5, color='red', label='Vitesse')
    # ax2.plot(mocap_time_axis[1:], moving_average(np.array(acceleration),10),alpha=0.5, color='green', label='Vitesse')
    
    # Configurer les étiquettes et la légende pour l'axe Y secondaire
    ax2.set_ylabel('Vitesse [m/s]', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')

    for time in mocap_ttl_times:
        plt.axvline(time,color="black")
    
    
    
    del whole_mocap_data

#%% Raster plot by trial ?

spike_times_all_units = []

for unit in unit_list:
    spike_times = sorter_results.get_unit_spike_train(unit_id=unit)/sampling_rate*Hz*s
    spike_times_all_units.append(spike_times)


"""
Rasterplot whole session
"""



# # Créer la figure et l'axe
# fig, ax = plt.subplots(figsize=(10, 6))

# # Parcourir chaque unité et créer des marqueurs pour les temps d'événements
# for i, unit_events in enumerate(spike_times_all_units):
#     ax.eventplot(unit_events, lineoffsets=i + 1, colors=f'C{i+1}', linewidths=2)

# # Ajuster les propriétés de l'axe
# ax.set_xlabel('Temps')
# ax.set_ylabel('Unités')
# ax.set_title('Raster Plot des Événements')
# ax.set_yticks(np.arange(1, len(spike_times_all_units) + 1))
# ax.set_yticklabels([f'Unité {i+1}' for i in range(len(spike_times_all_units))])
# ax.grid(True)

# # Afficher le raster plot
# plt.tight_layout()
# plt.show()



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
    
        speed = mocap_data['speed_back1']
        mocap_time_axis = (np.array(range(len(mocap_data)))/200+ttl_time*Hz)-mocap_delay/mocap_freq
    
    
        """
        Raster plot
        """
        # # Créer la figure et les axes pour les subplots
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # # Plot supérieur : plot de mocap_time_axis vs. speed
        # ax1.plot(mocap_time_axis, speed)
        # ax1.set_ylabel('Vitesse')
        # ax1.set_title(rf'Trial {trial}')
        
        # # Plot inférieur : raster plot des événements
        # for i, unit_events in enumerate(spike_times_all_units):
        #     events_to_plot = unit_events[(unit_events >= mocap_time_axis[0] * s) & (unit_events <= mocap_time_axis[-1] * s)]
        #     ax2.eventplot(events_to_plot, lineoffsets=i + 1, colors=f'C{i+1}', linewidths=2)
        
        # ax2.set_xlabel('Temps')
        # ax2.set_ylabel('Unités')
        # ax2.set_yticks(np.arange(1, len(spike_times_all_units) + 1))
        # ax2.set_yticklabels([f'Unité {i+1}' for i in unit_list])
        # ax2.grid(True)
        
        # # Ajuster les espacements et afficher le subplot
        # plt.tight_layout()
        # plt.show()
        
        
        """
        Heatmap
        """

        # # Convertir les temps d'événements entre les limites de temps en une matrice binaire
        # num_units = len(spike_times_all_units)
        
        # time_bin = 500 #ms
        # num_subdivisions = int((mocap_time_axis[-1] - mocap_time_axis[0])/(time_bin/1000))

        # num_time_steps = int((mocap_time_axis[-1] - mocap_time_axis[0]) * s * num_subdivisions) + 1

        # event_matrix = np.zeros((num_units, num_time_steps))
        
        # for i, unit_events in enumerate(spike_times_all_units):
        #     valid_events = unit_events[(unit_events >= mocap_time_axis[0] * s) & (unit_events <= mocap_time_axis[-1] * s)]
        #     time_indices = ((valid_events - mocap_time_axis[0]*s)* num_subdivisions).astype(int)
        #     event_matrix[i, time_indices] = 1
        
        # # Créer la figure et les axes pour les subplots
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # # Plot supérieur : plot de mocap_time_axis vs. speed
        # ax1.plot(mocap_time_axis, speed)
        # ax1.set_ylabel('Vitesse')
        # ax1.set_title('Plot de la Vitesse et Carte de Chaleur des Événements')
        
        # # Plot inférieur : carte de chaleur (heatmap) des événements entre les limites de temps
        # heatmap = ax2.imshow(event_matrix, cmap='BuPu', aspect='auto', extent=[mocap_time_axis[0], mocap_time_axis[-1], 0, num_units])
        
        # ax2.set_xlabel('Temps')
        # ax2.set_ylabel('Unités')
        # ax2.set_yticks(np.arange(1, num_units + 1))
        # ax2.set_yticklabels([f'Unité {i}' for i in unit_list])
        # ax2.grid(True)
        
        # # Ajouter une barre de couleur pour interpréter les valeurs
        # # cbar = plt.colorbar(heatmap, ax=ax2)
        # # cbar.set_label('Nombre d\'événements')
        
        # # Ajuster les espacements et afficher le subplot
        # plt.tight_layout()
        # plt.show() 
        
        # Convertir les temps d'événements entre les limites de temps en une matrice binaire
        num_units = len(spike_times_all_units)
        
        time_bin = 0.05  #s
        # Calculer le nombre de subdivisions en ajustant le calcul
        num_subdivisions = int((mocap_time_axis[-1] - mocap_time_axis[0]) / time_bin) + 1
        
        num_time_steps = int((mocap_time_axis[-1] - mocap_time_axis[0]) * num_subdivisions) + 1
        
        event_matrix = np.zeros((num_units, num_time_steps))
        
        for i, unit_events in enumerate(spike_times_all_units):
            valid_events = unit_events[(unit_events >= mocap_time_axis[0]*s) & (unit_events <= mocap_time_axis[-1]*s)]
            time_indices = ((valid_events - mocap_time_axis[0]*s) * num_subdivisions).astype(int)
            event_matrix[i, time_indices] = 1
        
        # Créer la figure et les axes pour les subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Plot supérieur : plot de mocap_time_axis vs. speed
        ax1.plot(mocap_time_axis, speed)
        ax1.set_ylabel('Vitesse')
        ax1.set_title('Plot de la Vitesse et Carte de Chaleur des Événements')
        
        # Plot inférieur : carte de chaleur (heatmap) des événements entre les limites de temps
        heatmap = ax2.imshow(event_matrix, cmap='BuPu', aspect='auto', extent=[mocap_time_axis[0], mocap_time_axis[-1], 0, num_units])
        
        ax2.set_xlabel('Temps')
        ax2.set_ylabel('Unités')
        ax2.set_yticks(np.arange(1, num_units + 1))
        ax2.set_yticklabels([f'Unité {i}' for i in range(1, num_units + 1)])

        
        # Ajouter une barre de couleur pour interpréter les valeurs
        # cbar = plt.colorbar(heatmap, ax=ax2)
        # cbar.set_label("Nombre d'événements")
        
        # Ajuster les espacements et afficher le subplot
        plt.tight_layout()
        plt.show()
        
        

        
        
        savefig_path = rf'{plots_path}/{animal}/Heatmap/Heatmap_{animal}_{mocap_session}_{trial}.png'
        Check_Save_Dir(os.path.dirname(savefig_path))
        plt.savefig(savefig_path)
    