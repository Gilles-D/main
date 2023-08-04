# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 14:36:03 2023

@author: Gil
"""

import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neo.core import SpikeTrain
from quantities import ms, s, Hz
from elephant.spike_train_generation import homogeneous_poisson_process, homogeneous_gamma_process
from elephant.statistics import mean_firing_rate
from elephant.statistics import time_histogram, instantaneous_rate
from elephant.kernels import GaussianKernel

import pickle
import time
import sys

#load intan class
sys.path.append(r"C:\Users\MOCAP\Documents\GitHub\main\In vivo Multi")

from intanutil.read_header import read_header
from intanutil.get_bytes_per_data_block import get_bytes_per_data_block
from intanutil.read_one_data_block import read_one_data_block
from intanutil.notch_filter import notch_filter
from intanutil.data_to_result import data_to_result

#Load MOCAP class
sys.path.append(r'C:\Users\MOCAP\Documents\GitHub\main\MOCAP\Analysis')
import MOCAP_analysis_class as MA


#%% Parameters
session_name = r'0026_01_08_allchan_allfiles'
spikesorting_results_path = r"D:\ePhy\SI_Data\spikesorting_results"
concatenated_signals_path = r'D:\ePhy\SI_Data\concatenated_signals'

# mocap_folder = rf'D:\ePhy\SI_Data\mocap_files'
# mocap_frequency = 200
# mocap_delay = 44/mocap_frequency

plot_format = 'png'

sites_location=[[0.0, 250.0],
  [0.0, 300.0],
  [0.0, 350.0],
  [0.0, 200.0],
  [0.0, 150.0],
  [0.0, 100.0],
  [0.0, 50.0],
  [0.0, 0.0],
  [43.3, 25.0],
  [43.3, 75.0],
  [43.3, 125.0],
  [43.3, 175.0],
  [43.3, 225.0],
  [43.3, 275.0],
  [43.3, 325.0],
  [43.3, 375.0]]

sampling_rate = 20000

#%%Functions

def read_data(filename):
    tic = time.time()
    fid = open(filename, 'rb')
    filesize = os.path.getsize(filename)
    header = read_header(fid)
    print('Found {} amplifier channel{}.'.format(
        header['num_amplifier_channels'], plural(header['num_amplifier_channels'])))
    print('Found {} auxiliary input channel{}.'.format(
        header['num_aux_input_channels'], plural(header['num_aux_input_channels'])))
    print('Found {} supply voltage channel{}.'.format(
        header['num_supply_voltage_channels'], plural(header['num_supply_voltage_channels'])))
    print('Found {} board ADC channel{}.'.format(
        header['num_board_adc_channels'], plural(header['num_board_adc_channels'])))
    print('Found {} board digital input channel{}.'.format(
        header['num_board_dig_in_channels'], plural(header['num_board_dig_in_channels'])))
    print('Found {} board digital output channel{}.'.format(
        header['num_board_dig_out_channels'], plural(header['num_board_dig_out_channels'])))
    print('Found {} temperature sensors channel{}.'.format(
        header['num_temp_sensor_channels'], plural(header['num_temp_sensor_channels'])))
    print('')
    # Determine how many samples the data file contains.
    bytes_per_block = get_bytes_per_data_block(header)
    # How many data blocks remain in this file?
    data_present = False
    bytes_remaining = filesize - fid.tell()
    if bytes_remaining > 0:
        data_present = True
    if bytes_remaining % bytes_per_block != 0:
        raise Exception(
            'Something is wrong with file size : should have a whole number of data blocks')
    num_data_blocks = int(bytes_remaining / bytes_per_block)
    num_amplifier_samples = header['num_samples_per_data_block'] * \
        num_data_blocks
    num_aux_input_samples = int(
        (header['num_samples_per_data_block'] / 4) * num_data_blocks)
    num_supply_voltage_samples = 1 * num_data_blocks
    num_board_adc_samples = header['num_samples_per_data_block'] * \
        num_data_blocks
    num_board_dig_in_samples = header['num_samples_per_data_block'] * \
        num_data_blocks
    num_board_dig_out_samples = header['num_samples_per_data_block'] * \
        num_data_blocks
    record_time = num_amplifier_samples / header['sample_rate']
    if data_present:
        print('File contains {:0.3f} seconds of data.  Amplifiers were sampled at {:0.2f} kS/s.'.format(
            record_time, header['sample_rate'] / 1000))
    else:
        print('Header file contains no data.  Amplifiers were sampled at {:0.2f} kS/s.'.format(
            header['sample_rate'] / 1000))
    if data_present:
        # Pre-allocate memory for data.
        print('')
        print('Allocating memory for data...')
        data = {}
        if (header['version']['major'] == 1 and header['version']['minor'] >= 2) or (header['version']['major'] > 1):
            data['t_amplifier'] = np.zeros(
                num_amplifier_samples, dtype=np.int_)
        else:
            data['t_amplifier'] = np.zeros(
                num_amplifier_samples, dtype=np.uint)
        data['amplifier_data'] = np.zeros(
            [header['num_amplifier_channels'], num_amplifier_samples], dtype=np.uint)
        data['aux_input_data'] = np.zeros(
            [header['num_aux_input_channels'], num_aux_input_samples], dtype=np.uint)
        data['supply_voltage_data'] = np.zeros(
            [header['num_supply_voltage_channels'], num_supply_voltage_samples], dtype=np.uint)
        data['temp_sensor_data'] = np.zeros(
            [header['num_temp_sensor_channels'], num_supply_voltage_samples], dtype=np.uint)
        data['board_adc_data'] = np.zeros(
            [header['num_board_adc_channels'], num_board_adc_samples], dtype=np.uint)
        # by default, this script interprets digital events (digital inputs and outputs) as booleans
        # if unsigned int values are preferred(0 for False, 1 for True), replace the 'dtype=np.bool_' argument with 'dtype=np.uint' as shown
        # the commented line below illustrates this for digital input data; the same can be done for digital out
        #data['board_dig_in_data'] = np.zeros([header['num_board_dig_in_channels'], num_board_dig_in_samples], dtype=np.uint)
        data['board_dig_in_data'] = np.zeros(
            [header['num_board_dig_in_channels'], num_board_dig_in_samples], dtype=np.bool_)
        data['board_dig_in_raw'] = np.zeros(
            num_board_dig_in_samples, dtype=np.uint)
        data['board_dig_out_data'] = np.zeros(
            [header['num_board_dig_out_channels'], num_board_dig_out_samples], dtype=np.bool_)
        data['board_dig_out_raw'] = np.zeros(
            num_board_dig_out_samples, dtype=np.uint)
        # Read sampled data from file.
        print('Reading data from file...')
        # Initialize indices used in looping
        indices = {}
        indices['amplifier'] = 0
        indices['aux_input'] = 0
        indices['supply_voltage'] = 0
        indices['board_adc'] = 0
        indices['board_dig_in'] = 0
        indices['board_dig_out'] = 0
        print_increment = 10
        percent_done = print_increment
        for i in range(num_data_blocks):
            read_one_data_block(data, header, indices, fid)
            # Increment indices
            indices['amplifier'] += header['num_samples_per_data_block']
            indices['aux_input'] += int(
                header['num_samples_per_data_block'] / 4)
            indices['supply_voltage'] += 1
            indices['board_adc'] += header['num_samples_per_data_block']
            indices['board_dig_in'] += header['num_samples_per_data_block']
            indices['board_dig_out'] += header['num_samples_per_data_block']
            fraction_done = 100 * (1.0 * i / num_data_blocks)
            if fraction_done >= percent_done:
                print('{}% done...'.format(percent_done))
                percent_done = percent_done + print_increment
        # Make sure we have read exactly the right amount of data.
        bytes_remaining = filesize - fid.tell()
        if bytes_remaining != 0:
            raise Exception('Error: End of file not reached.')
    # Close data file.
    fid.close()
    if (data_present):
        print('Parsing data...')
        # Extract digital input channels to separate variables.
        for i in range(header['num_board_dig_in_channels']):
            data['board_dig_in_data'][i, :] = np.not_equal(np.bitwise_and(
                data['board_dig_in_raw'], (1 << header['board_dig_in_channels'][i]['native_order'])), 0)
        # Extract digital output channels to separate variables.
        for i in range(header['num_board_dig_out_channels']):
            data['board_dig_out_data'][i, :] = np.not_equal(np.bitwise_and(
                data['board_dig_out_raw'], (1 << header['board_dig_out_channels'][i]['native_order'])), 0)
        # Scale voltage levels appropriately.
        data['amplifier_data'] = np.multiply(
            0.195, (data['amplifier_data'].astype(np.int32) - 32768))      # units = microvolts
        data['aux_input_data'] = np.multiply(
            37.4e-6, data['aux_input_data'])               # units = volts
        data['supply_voltage_data'] = np.multiply(
            74.8e-6, data['supply_voltage_data'])     # units = volts
        if header['eval_board_mode'] == 1:
            data['board_adc_data'] = np.multiply(
                152.59e-6, (data['board_adc_data'].astype(np.int32) - 32768))  # units = volts
        elif header['eval_board_mode'] == 13:
            data['board_adc_data'] = np.multiply(
                312.5e-6, (data['board_adc_data'].astype(np.int32) - 32768))  # units = volts
        else:
            data['board_adc_data'] = np.multiply(
                50.354e-6, data['board_adc_data'])           # units = volts
        data['temp_sensor_data'] = np.multiply(
            0.01, data['temp_sensor_data'])               # units = deg C
        # Check for gaps in timestamps.
        num_gaps = np.sum(np.not_equal(
            data['t_amplifier'][1:]-data['t_amplifier'][:-1], 1))
        if num_gaps == 0:
            print('No missing timestamps in data.')
        else:
            print('Warning: {0} gaps in timestamp data found.  Time scale will not be uniform!'.format(
                num_gaps))
        # Scale time steps (units = seconds).
        data['t_amplifier'] = data['t_amplifier'] / header['sample_rate']
        data['t_aux_input'] = data['t_amplifier'][range(
            0, len(data['t_amplifier']), 4)]
        data['t_supply_voltage'] = data['t_amplifier'][range(
            0, len(data['t_amplifier']), header['num_samples_per_data_block'])]
        data['t_board_adc'] = data['t_amplifier']
        data['t_dig'] = data['t_amplifier']
        data['t_temp_sensor'] = data['t_supply_voltage']
        # If the software notch filter was selected during the recording, apply the
        # same notch filter to amplifier data here.
        if header['notch_filter_frequency'] > 0 and header['version']['major'] < 3:
            print('Applying notch filter...')
            print_increment = 10
            percent_done = print_increment
            for i in range(header['num_amplifier_channels']):
                data['amplifier_data'][i, :] = notch_filter(
                    data['amplifier_data'][i, :], header['sample_rate'], header['notch_filter_frequency'], 10)
                fraction_done = 100 * (i / header['num_amplifier_channels'])
                if fraction_done >= percent_done:
                    print('{}% done...'.format(percent_done))
                    percent_done += print_increment
    else:
        data = []
    # Move variables to result struct.
    result = data_to_result(header, data, data_present)
    print('Done!  Elapsed time: {0:0.1f} seconds'.format(time.time() - tic))
    return result
def plural(n):
    if n == 1:
        return ''
    else:
        return 's'

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


def Get_spikes(session_name, spikesorting_results_path):
    """
    Retrieves spike times from XLSX files in a given session directory.

    Args:
        session_name (str): Name of the session directory.
        spikesorting_results_path (str): Path to the spikesorting results directory.

    Returns:
        dict: Dictionary containing unit names, spike times array, and spike trains.

    """
    print('Reading spikes csv files...')

    # Set the path to the spike times directory
    spike_times_path = rf'{spikesorting_results_path}\{session_name}\spikes'

    # List all units
    unit_list = [file_name.split(".")[0] for file_name in os.listdir(spike_times_path)]

    # Use glob.glob() to get a list of XLSX files in the directory
    file_paths = glob.glob(os.path.join(spike_times_path, '*.xlsx'))

    # Create lists to store the spike times arrays and spike trains
    spike_times_array, elephant_spiketrains = [], []

    # Loop through the XLSX files
    for file_path in file_paths:
        # Load the XLSX file into a pandas DataFrame and retrieve the second column as spike times
        spike_times = np.array(pd.read_excel(file_path).iloc[:, 1])
        spike_times_array.append(spike_times)

        # Calculate t_stop as the maximum spike time plus 1
        t_stop = max(spike_times) + 1

        # Create a spike train using the Elephant library
        elephant_spiketrains.append(SpikeTrain(spike_times * s, t_stop=t_stop,sampling_rate=10*Hz))

    # Create a dictionary containing the unit names, spike times arrays, and spike trains
    spike_times_dict = {'Units': unit_list, 'spike times': spike_times_array, 'spiketrains': elephant_spiketrains}

    print('Done')

    return spike_times_dict


def Get_recordings_info(session_name, concatenated_signals_path, spikesorting_results_path):
    save_path = rf'{spikesorting_results_path}/{session_name}/recordings_info.pickle'
    if os.path.exists(save_path):
        print("Recordings info file exists")
        print("Loading info file...")
        recordings_info = pickle.load(open(save_path, "rb"))
    else:
        print("Recordings info file does not exist")
        print("Getting info...")
        # Read the metadata file created during concatenation
        path = rf'{concatenated_signals_path}/{session_name}/'
        metadata = pickle.load(open(rf"{path}/metadata.pickle", "rb"))
       
        # Loop over intan files
        recordings_list = metadata['recordings_files']
        # RHD file reading
        multi_recordings, recordings_lengths, multi_stim_idx, multi_frame_idx, frame_start_delay = [], [], [], [], []
        
        # Concatenate recordings
        for record in recordings_list:
            reader = read_data(record)
            signal = reader['amplifier_data']
            recordings_lengths.append(len(signal[0]))
            multi_recordings.append(signal)
            
            stim_idx = reader['board_dig_in_data'][0]  # Digital data for stim of the file
            multi_stim_idx.append(stim_idx)  # Digital data for stim of all the files
            
            frame_idx = reader['board_dig_in_data'][1]  # Get digital data for mocap ttl
            multi_frame_idx.append(frame_idx)  # Digital data for mocap ttl of all the files
            
        anaglog_signal_concatenated = np.hstack(multi_recordings)  # Signal concatenated from all the files
        digital_stim_signal_concatenated = np.hstack(multi_stim_idx)  # Digital data for stim concatenated from all the files
        digital_mocap_signal_concatenated = np.hstack(multi_frame_idx)
        
        # Get sampling freq
        sampling_rate = reader['frequency_parameters']['amplifier_sample_rate']
        
        recordings_lengths_cumsum = np.cumsum(np.array(recordings_lengths) / sampling_rate)
                                              
        # Return: recording length, recording length cumsum, digital signals 1 and 2 (in full or logical?)
        # Save them in a pickle
        
        recordings_info = {
            'recordings_length': recordings_lengths,
            'recordings_length_cumsum': recordings_lengths_cumsum,
            'sampling_rate': sampling_rate,
            'digital_stim_signal_concatenated': digital_stim_signal_concatenated,
            'digital_mocap_signal_concatenated': digital_mocap_signal_concatenated
        }
        
        pickle.dump(recordings_info, open(save_path, "wb"))
        
    print('Done')
    return recordings_info


def plot_waveform_old(session_name, spikesorting_results_path, sites_location, unit,save=True):
    file_pattern = rf"Unit_{unit} *"
    matching_files = glob.glob(rf"{spikesorting_results_path}/{session_name}/waveforms/{file_pattern}")
    
    print(rf"{spikesorting_results_path}/{session_name}/waveforms/{file_pattern}")
    
    if len(matching_files) > 0:
        print("Matching file(s) found:")
        for file_path in matching_files:
            print(file_path)
            
            df = pd.read_excel(file_path)
            
            fig1 = plt.figure(figsize=(10, 12))
            ax1 = fig1.add_subplot(111)
            
            fig1.suptitle(rf'Average Waveform Unit # {unit}')
            ax1.set_xlabel('Probe location (micrometers)')
            ax1.set_ylabel('Probe location (micrometers)')
            
            for loc, prob_loc in enumerate(sites_location):
                x_offset, y_offset = prob_loc[0], prob_loc[1]
                base_x = np.linspace(-15, 15, num=len(df.iloc[:, loc]))  # Basic x-array for plot, centered
                # clust_color = 'C{}'.format(cluster)
                
                wave = df.iloc[:, loc] * 10 + y_offset
                ax1.plot(base_x + 2 * x_offset, wave)
                # ax1.fill_between(base_x + 2 * x_offset, wave - wf_rms[cluster + delta], wave + wf_rms[cluster + delta], alpha=wf_alpha)
                
            plt.show()
            if save==True:
                save_path = rf"{spikesorting_results_path}/{session_name}/plots/waveforms/"
                Check_Save_Dir(save_path)
                plt.savefig(rf"{save_path}/Units {unit}.{plot_format}")
    else:
        print("No matching file found")
        
    return


def plot_waveform(session_name, spikesorting_results_path, sites_location, unit, save=True):
    file_pattern = rf"Unit_{unit} *"
    matching_files = glob.glob(rf"{spikesorting_results_path}/{session_name}/waveforms/{file_pattern}")
    
    print(rf"{spikesorting_results_path}/{session_name}/waveforms/{file_pattern}")
    
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
                plt.savefig(rf"{save_path}/Units {unit}.{plot_format}")
    else:
        print("No matching file found")
        
    return



def Mocap_start_ttl_detection(mocap_starts_idx, sampling_rate):
    # Calculate the difference between consecutive elements
    diff_indices = np.diff(mocap_starts_idx)
    phase_indices = np.where(diff_indices != 1)[0] - 1
    start_indexes = mocap_starts_idx[phase_indices]
    start_times = start_indexes / sampling_rate
    
    return start_times


def create_time_periods(times):
    time_periods = []
    if len(times) < 2:
        return time_periods
    t_start = times[0]
    for i in range(1, len(times)):
        t_stop = times[i]
        time_periods.append((t_start, t_stop))
        t_start = t_stop
    return time_periods


def separate_events_by_period(events, time_periods):
    num_periods = len(time_periods)
    event_periods = []
    
    for i in range(num_periods):
        t_start, t_stop = time_periods[i]
        period_events = events[(events >= t_start) & (events < t_stop)]
        event_periods.append(period_events - t_start)
        
    event_periods_dict = {}
    
    for i, array in enumerate(event_periods):
        event_periods_dict[rf'Mocap_Session_{i + 1}'] = array
        
    return event_periods_dict


def plot_heatmap(spike_times, bin_size):

    event_times_list = spike_times.values()
    event_session_list = spike_times.keys()
    
    # Set the bin size and create the time bins
    max_time = max(max(times) for times in event_times_list)
    
    bins = np.arange(0, max_time + bin_size, bin_size)
    
    # Compute the histograms of event counts for each row
    histograms = []
    for event_times in event_times_list:
        counts, _ = np.histogram(event_times, bins)
        histograms.append(counts)
        
    # Check if any histogram has fewer than two bins
    if any(len(counts) < 2 for counts in histograms):
        raise ValueError("Insufficient data for heatmap visualization.")
        
    # Create a 2D array from the histograms for heatmap visualization
    heatmap = np.array(histograms)
    
    # Plot the heatmap
    plt.figure()
    plt.imshow(heatmap, cmap='hot', aspect='auto')
    plt.colorbar(label='Event Count')
    plt.xlabel('Time Bin')
    plt.ylabel('Event session')
    
    # Set the y-axis tick labels to the event_session_list items
    plt.yticks(range(len(event_session_list)), event_session_list)
    
    plt.title('Event Heatmap in Line Plot')
    plt.show()




def plot_heatmap_start_fixed(spike_times, bin_size):

    event_times_list = spike_times.values()
    event_session_list = spike_times.keys()
    
    # Set the bin size and create the time bins
    max_time = max(max(times) for times in event_times_list)
    
    bins = np.arange(-10, max_time + bin_size, bin_size)
    
    # Compute the histograms of event counts for each row
    histograms = []
    for event_times in event_times_list:
        counts, _ = np.histogram(event_times, bins)
        histograms.append(counts)
        
    # Check if any histogram has fewer than two bins
    if any(len(counts) < 2 for counts in histograms):
        raise ValueError("Insufficient data for heatmap visualization.")
        
    # Create a 2D array from the histograms for heatmap visualization
    heatmap = np.array(histograms)
    
    # Plot the heatmap

    plt.imshow(heatmap, cmap='hot', aspect='auto')
    plt.colorbar(label='Event Count')
    plt.xlabel('Time Bin')
    plt.ylabel('Event session')
    
    # Set the y-axis tick labels to the event_session_list items
    plt.yticks(range(len(event_session_list)), event_session_list)

    plt.show()



def find_start_stop_obstacle(mocap_folder,session_name,mocap_frequency=200):
    list_mocap_files = glob.glob(os.path.join(rf"{mocap_folder}/{session_name}", '*.csv'))
    mocap_data_dict={}

    for file in list_mocap_files:
        print(file)
        data_MOCAP = MA.MOCAP_file(file)
        session_idx = int(file.split('\\')[-1].split('.')[0].split('_')[-1])
        session = rf"Mocap_Session_{session_idx}"
        
        back_coord = data_MOCAP.coord(f"{data_MOCAP.subject()}:Back1")
        start_x = -np.nanmedian(data_MOCAP.coord(f"{data_MOCAP.subject()}:Platform1")[1])
        stop_x = -np.nanmedian(data_MOCAP.coord(f"{data_MOCAP.subject()}:Platform2")[1])
           
       
        start_frame = np.where(-back_coord[1] > start_x)[0][0]
        start_time = start_frame/mocap_frequency
        
        stop_frame = np.where(-back_coord[1] > stop_x)[0][0]
        stop_time = stop_frame/mocap_frequency
        
        try:
            obstacle_x = -np.nanmedian(data_MOCAP.coord(f"{data_MOCAP.subject()}:Obstacle1")[1])
            obstacle_frame = np.where(-back_coord[1] > obstacle_x)[0][0]
            obstacle_time = obstacle_frame/mocap_frequency
            
        except:
            print("no obstacle")
            obstacle_time=np.nan
        
        
        
        # plt.figure()
        # plt.title(file)
        # plt.plot(-back_coord[1],back_coord[2])
        # plt.axvline(start_x)
        # plt.axvline(stop_x)
        # plt.axvline(obstacle_x)
        
        
        file_dict = {
            "start_time": start_time,
            "stop_time": stop_time,
            "obstacle_time": obstacle_time
        }
        
        mocap_data_dict[session] = file_dict
    return mocap_data_dict





#%% Loadings
recordings_info = Get_recordings_info(session_name,concatenated_signals_path,spikesorting_results_path)
with open(fr'{concatenated_signals_path}/{session_name}/metadata.pickle', 'rb') as handle:
    metadata = pickle.load(handle)

spike_times_dict = Get_spikes(session_name,spikesorting_results_path)

# plot_waveform(session_name,spikesorting_results_path,sites_location,unit=8)



#%% Split by MOCAP TTL
mocap_start_stop_dict=find_start_stop_obstacle(mocap_folder,session_name)

if not os.path.exists(fr'{spikesorting_results_path}/{session_name}/spike_times_dict.pickle') or os.path.exists(fr'{spikesorting_results_path}/{session_name}/spike_times_by_mocap_session.pickle'):
    # Detect Mocap start signals
    mocap_starts_idx = (np.where(recordings_info['digital_mocap_signal_concatenated'] == True))[0]
    mocap_starts_times = Mocap_start_ttl_detection(mocap_starts_idx, sampling_rate)
    mocap_session_periods = create_time_periods(mocap_starts_times)
    
    # Split spike times by mocap trial
    spike_times_split_list = []
    spike_times_by_mocap_session = {}
    
    for unit in spike_times_dict['Units']:
        spike_times = spike_times_dict['spike times'][spike_times_dict['Units'].index(unit)]
        spike_times_by_session = separate_events_by_period(spike_times, mocap_session_periods)
        spike_times_split_list.append(spike_times_by_session)
    
    for i, dictionary in enumerate(spike_times_split_list):
        unit = spike_times_dict['Units'][i]
        spike_times_by_mocap_session[unit] = dictionary
        
    del spike_times_split_list, spike_times, spike_times_by_session, mocap_starts_idx, mocap_starts_times
    
    #Save the dictionnaries
    with open(fr'{spikesorting_results_path}/{session_name}/spike_times_dict.pickle', 'wb') as handle:
        pickle.dump(spike_times_dict, handle)
    
    with open(fr'{spikesorting_results_path}/{session_name}/spike_times_by_mocap_session.pickle', 'wb') as handle:
        pickle.dump(spike_times_by_mocap_session, handle)

else:
    print('Split spiketimes pickle file already exists. Loading them...')
    with open(fr'{spikesorting_results_path}/{session_name}/spike_times_dict.pickle', 'rb') as handle:
        spike_times_dict = pickle.load(handle)
        
    with open(fr'{spikesorting_results_path}/{session_name}/spike_times_by_mocap_session.pickle', 'rb') as handle:
        spike_times_by_mocap_session = pickle.load(handle)



#%% Figure 1 : Whole SpikeTrain Analysis
print('Figure 1 - Elephant Spike Train Analysis')

# pose_list= [1563,1112,1082,1667,850,765,722,700,1338,1361,1700]


for unit in spike_times_dict['Units']:
    print(unit)
    
    plot_waveform(session_name, spikesorting_results_path, sites_location, unit.split(' ')[0].split('_')[1])
    
    
    spiketrain = spike_times_dict['spiketrains'][spike_times_dict['Units'].index(unit)]
    

    
    plt.figure()
    
    # Compute the time histogram and rate histogram
    histogram_count = time_histogram([spiketrain], 0.1 * s)
    histogram_rate = time_histogram([spiketrain], 0.1 * s, output='rate')
    
    # Compute the instantaneous rate
    inst_rate = instantaneous_rate(spiketrain, sampling_period=5 * ms)
    
    recordings_cumsum = recordings_info['recordings_length_cumsum']
    for i in recordings_cumsum:
        plt.axvline(i)
    
    #Baseline session = first file
    # plt.axvspan(0, recordings_cumsum[0],color='green',alpha=0.3)
    
    # Compute and print the mean firing rate
    mean_fr = mean_firing_rate(spiketrain)
    baseline_mean_fr = mean_firing_rate(spiketrain,t_start=0*s,t_stop=recordings_cumsum[0]*s)
    
    print(f"The mean firing rate of unit {unit} is {mean_fr} on the whole session and {baseline_mean_fr} for the baseline")
    
    
    i=0
    # for session, start_stop_dict in mocap_start_stop_dict.items():
    #     start_time = start_stop_dict['start_time']+mocap_session_periods[i][0]-mocap_delay
    #     stop_time=start_stop_dict['stop_time']+mocap_session_periods[i][0]-mocap_delay
        
    #     obstacle_time = start_stop_dict['obstacle_time']+mocap_session_periods[i][0]-mocap_delay
        
    #     plt.axvline(mocap_session_periods[i][0]-mocap_delay,color='grey',linestyle='dashed')
        
    #     plt.axvspan(start_time, stop_time, color='red',alpha=0.3)
    #     plt.axvline(obstacle_time,color='black')
    #     i=i+1
    
    # for idx,i in enumerate(pose_list):
    #     pose_time = (i/mocap_frequency)+mocap_delay+mocap_session_periods[idx][0]

    #     plt.axvspan(pose_time-0.5, pose_time+0.5, color='blue',alpha=0.3)
    

    # Plot the time histogram (rate)
    # plt.bar(histogram_rate.times, histogram_rate.magnitude.flatten(), width=histogram_rate.sampling_period,
    #         align='edge', alpha=0.3, label='time histogram (rate)', color='black')
    
    
    # Plot the instantaneous rate
    plt.plot(inst_rate.times.rescale(s), inst_rate.rescale(histogram_rate.dimensionality).magnitude.flatten(),
             label='instantaneous rate')
    
    # Set the axis labels and legend
    plt.xlabel('time [{}]'.format(spiketrain.times.dimensionality.latex))
    plt.ylabel('firing rate [{}]'.format(histogram_rate.dimensionality.latex))
    
    plt.xlim(spiketrain.t_start, spiketrain.t_stop)
    
    plt.legend()
    plt.title(rf'Spiketrain {unit}')
    plt.show()
    
    # Create the save directory
    savefig_folder = rf'{spikesorting_results_path}/{session_name}/plots/spiking_analysis/'
    Check_Save_Dir(savefig_folder)
    
    # Save the figure
    plt.savefig(rf"{savefig_folder}/Figure 1 - Elephant Spike Train Analysis - {unit}.{plot_format}",dpi=900)



    

#%% Figure 2 - Heatmap spiking by unit, by mocap session

# Create the save directory
savefig_folder = rf'{spikesorting_results_path}/{session_name}/plots/heatmaps/'
Check_Save_Dir(savefig_folder)

for unit in spike_times_dict['Units']:
    print(unit)
    spike_times = spike_times_by_mocap_session[unit]
    plot_heatmap(spike_times,bin_size = 0.5)
    plt.savefig(rf"{savefig_folder}{unit}.png")
    
    
    # t_stop = None

    # for key, value in spike_times.items():
    #     if t_stop is None:
    #         t_stop = np.max(value)
    #     else:
    #         t_stop = max(t_stop, np.max(value))
    
    # spiketrains=[]
    
    # # plt.figure()
    # # plt.title(rf'Unit # {unit}')
    
    # for i,session in enumerate(spike_times.keys()):
        
    #     spiketrain = SpikeTrain(spike_times[session]*s, t_stop)
    #     inst_rate = instantaneous_rate(spiketrain, 5*ms)
        

#%% 3- Raster + Psth start
# Create the save directory
savefig_folder = rf'{spikesorting_results_path}/{session_name}/plots/Raster-PSTH/'
Check_Save_Dir(savefig_folder)

with open(fr'{spikesorting_results_path}/{session_name}/spike_times_by_mocap_session.pickle', 'rb') as handle:
    spike_times_by_mocap_session = pickle.load(handle)

for unit in spike_times_dict['Units']:
    print(unit)
    spike_times = spike_times_by_mocap_session[unit]
    for session, start_stop_dict in mocap_start_stop_dict.items():
        start_time = start_stop_dict['start_time']
        if session in spike_times:
            spike_times[session] -= start_time
    
    # Créer une figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    
    # Sous-graphique supérieur pour les spike_times
    ax1.set_title(rf'Unit {unit}')
    for i, (session, session_times) in enumerate(spike_times.items()):
        color = 'black' if session_times[i] < 0 else 'red'
        ax1.eventplot([session_times], lineoffsets=i + 0.5, linelengths=0.5, color=color)
    ax1.axvline(0, color='red')
    ax1.set_xlim(-10, 15)
    ax1.set_ylim(0, 11)
    
    # Sous-graphique inférieur pour le PSTH
    ax2.set_xlabel('Temps (s)')
    ax2.set_ylabel('Nombre de spikes')
    ax2.set_title('PSTH des 11 premières sessions')
    
    session_times = np.concatenate(list(spike_times.values())[:11])
    bin_size = 0.01  # Taille de chaque bin en secondes
    num_bins = int((max(session_times) - min(session_times)) / bin_size)
    
    hist, bin_edges = np.histogram(session_times, bins=num_bins, range=(min(session_times), max(session_times)))
    
    bar_color = np.where(bin_edges[:-1] < 0, 'black', 'red')
    ax2.bar(bin_edges[:-1], hist, width=bin_size, color=bar_color)
    
    # Ajuster l'espace entre les deux sous-graphiques
    plt.subplots_adjust(hspace=0.3)
    
    # Enregistrer la figure
    plt.savefig(rf"{savefig_folder}{unit}.png")
    
    # Afficher la figure
    plt.show()

#%%Raster + PSTH obstacle
# Create the save directory
savefig_folder = rf'{spikesorting_results_path}/{session_name}/plots/Raster-PSTH-obstacle/'
Check_Save_Dir(savefig_folder)

with open(fr'{spikesorting_results_path}/{session_name}/spike_times_by_mocap_session.pickle', 'rb') as handle:
    spike_times_by_mocap_session = pickle.load(handle)

for unit in spike_times_dict['Units']:
    print(unit)
    spike_times = spike_times_by_mocap_session[unit]
    for session, start_stop_dict in mocap_start_stop_dict.items():
        obstacle_time = start_stop_dict['obstacle_time']
        if session in spike_times:
            spike_times[session] -= obstacle_time
    
    # Créer une figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    
    # Sous-graphique supérieur pour les spike_times
    ax1.set_title(rf'Unit {unit}')
    for i, (session, session_times) in enumerate(spike_times.items()):
        color = 'black' if session_times[i] < 0 else 'red'
        ax1.eventplot([session_times], lineoffsets=i + 0.5, linelengths=0.5, color=color)
    ax1.axvline(0, color='red')
    ax1.set_xlim(-10, 15)
    ax1.set_ylim(7, 11)
    
    # Sous-graphique inférieur pour le PSTH
    ax2.set_xlabel('Temps (s)')
    ax2.set_ylabel('Nombre de spikes')
    ax2.set_title('PSTH des sessions obstacles')
    
    session_times = np.concatenate(list(spike_times.values())[7:10])
    bin_size = 0.01  # Taille de chaque bin en secondes
    num_bins = int((max(session_times) - min(session_times)) / bin_size)
    
    hist, bin_edges = np.histogram(session_times, bins=num_bins, range=(min(session_times), max(session_times)))
    
    bar_color = np.where(bin_edges[:-1] < 0, 'black', 'red')
    ax2.bar(bin_edges[:-1], hist, width=bin_size, color=bar_color)
    
    # Ajuster l'espace entre les deux sous-graphiques
    plt.subplots_adjust(hspace=0.3)
    
    # Enregistrer la figure
    plt.savefig(rf"{savefig_folder}{unit}.png")
    
    # Afficher la figure
    plt.show()





    # spike_times = spike_times_by_mocap_session[unit]
    # for session, start_stop_dict in mocap_start_stop_dict.items():
    #     start_time = start_stop_dict['start_time']
    #     if session in spike_times:
    #         spike_times[session] -= start_time
    
    # # Créer une figure avec deux sous-graphiques
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    
    # # Sous-graphique supérieur pour les spike_times
    # ax1.set_title(rf'Unit {unit}')
    # ax1.eventplot(spike_times.values())
    # ax1.axvline(0)
    # ax1.set_xlim(-10, 15)
    # ax1.set_ylim(0, 10)
    
    # # Sous-graphique inférieur pour le PSTH
    # ax2.set_xlabel('Temps (s)')
    # ax2.set_ylabel('Nombre de spikes')
    # ax2.set_title('PSTH des 10 premières sessions')
    
    # ax2.axvline(0)
    
    # session_times = np.concatenate(list(spike_times.values())[:10])
    # bin_size = 0.01  # Taille de chaque bin en secondes
    # num_bins = int((max(session_times) - min(session_times)) / bin_size)
    
    # hist, bin_edges = np.histogram(session_times, bins=num_bins, range=(min(session_times), max(session_times)))
    
    # ax2.bar(bin_edges[:-1], hist, width=bin_size)
    
    # # Ajuster l'espace entre les deux sous-graphiques
    # plt.subplots_adjust(hspace=0.3)
    
    # # Enregistrer la figure
    # plt.savefig(rf"{savefig_folder}Raster_PSTH_{unit}.png")
    
    # # Afficher la figure
    # plt.show()
    
            
    


