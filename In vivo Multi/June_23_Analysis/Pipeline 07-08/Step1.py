# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:59:35 2023

@author: MOCAP
"""
#%% modules
import spikeinterface as si
import spikeinterface.extractors as se 
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.comparison as sc
import spikeinterface.exporters as sexp
import spikeinterface.widgets as sw

import os
import sys
import time

import probeinterface as pi
from probeinterface.plotting import plot_probe

import warnings
warnings.simplefilter("ignore")


import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


from viziphant.statistics import plot_time_histogram
from viziphant.rasterplot import rasterplot_rates
from elephant.statistics import time_histogram
from neo.core import SpikeTrain
from quantities import s, ms
import pandas as pd

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

def list_recording_files(path):
    """
    List all recording files (.rhd) in the specified directory and its subdirectories.
    
    Parameters:
        path (str): The directory path to search for recording files.
        
    Returns:
        list: A list of paths to all recording files found.
    """
    
    import glob
    fichiers = [fichier for fichier in glob.iglob(path + '/**/*', recursive=True) if not os.path.isdir(fichier) and fichier.endswith('.rhd')]
    
    return fichiers


def TTL_detection(TTL_starts_idx, sampling_rate):
    """
    Detects start indexes and times of TTL pulses.

    Args:
        TTL_starts_idx (numpy.ndarray): A 1D numpy array containing the start indexes of TTL pulses.
        sampling_rate (float): The sampling rate of the recording.

    Returns:
        tuple: A tuple containing two arrays:
            - start_indexes (numpy.ndarray): A 1D numpy array containing the start indexes of each phase of TTL pulses.
            - start_times (numpy.ndarray): A 1D numpy array containing the start times (in seconds) of each phase of TTL pulses.
    """
    # Calculate the difference between consecutive elements
    diff_indices = np.diff(TTL_starts_idx)
    phase_indices = np.where(diff_indices != 1)[0]
    
    phase_end_indices = np.where(diff_indices != 1)[0]
    
    start_indexes = TTL_starts_idx[phase_indices] #fin de ttl
    
    start_indexes = np.insert(start_indexes, 0, TTL_starts_idx[0])
    start_times = start_indexes / sampling_rate

    return start_indexes, start_times



def trouver_changements_indices(tableau):
    diff = np.diff(tableau.astype(int))
    return np.where(diff != 0)[0] + 1
   




def Get_recordings_info(session_name, concatenated_signals_path, recordings_list,freq_min,freq_max,probe_path,excluded_sites):
    #load intan class
    sys.path.append(r"C:\Users\MOCAP\Documents\GitHub\main\In vivo Multi")

    from intanutil.read_header import read_header
    from intanutil.get_bytes_per_data_block import get_bytes_per_data_block
    from intanutil.read_one_data_block import read_one_data_block
    from intanutil.notch_filter import notch_filter
    from intanutil.data_to_result import data_to_result
    
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


    save_dir = rf'{concatenated_signals_path}/{session_name}'
    save_path= rf"{save_dir}/recordings_info.pickle"
    
    if os.path.exists(save_path):
        print("Recordings info file exists")
        print("Loading info file...")
        recordings_info = pickle.load(open(save_path, "rb"))
    else:
        print("Recordings info file does not exist")
        print("Getting info...")

        
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
        
                
        # Get sampling freq
        sampling_rate = reader['frequency_parameters']['amplifier_sample_rate']
        
        anaglog_signal_concatenated = np.hstack(multi_recordings)  # Signal concatenated from all the files
        digital_stim_signal_concatenated = np.hstack(multi_stim_idx)  # Digital data for stim concatenated from all the files
        digital_mocap_signal_concatenated = np.hstack(multi_frame_idx)
        
        try:
            # stim_ttl_on = TTL_detection((np.where(digital_stim_signal_concatenated == True))[0], sampling_rate)
            stim_ttl_on_off = trouver_changements_indices(digital_stim_signal_concatenated)
            
        except:
            print("No stim ttl")
            stim_ttl_on_off=np.array([])
        
        try:
            # mocap_ttl_on = TTL_detection((np.where(digital_mocap_signal_concatenated == True))[0], sampling_rate)
            mocap_ttl_on_off  = trouver_changements_indices(digital_mocap_signal_concatenated)
        except:
            print("No video ttl")
            mocap_ttl_on_off=np.array([])

        
        recordings_lengths_cumsum = np.cumsum(np.array(recordings_lengths) / sampling_rate)
                                              
        # Return: recording length, recording length cumsum, digital signals 1 and 2 (in full or logical?)
        # Save them in a pickle
        
        recordings_info = {
            "recordings_files" : recordings_list,
            "excluded_sites" : excluded_sites,
            "freq_min" : freq_min,
            "freq_max" : freq_max,
            "probe_path" : probe_path,
            'recordings_length': recordings_lengths,
            'recordings_length_cumsum': recordings_lengths_cumsum,
            'sampling_rate': sampling_rate,
            'digital_stim_signal_concatenated': digital_stim_signal_concatenated,
            'digital_mocap_signal_concatenated': digital_mocap_signal_concatenated,
            'stim_ttl_on':stim_ttl_on_off,
            'mocap_ttl_on':mocap_ttl_on_off
        }

    print('Done')
    return recordings_info





def concatenate_preprocessing(recordings,saving_dir,saving_name,probe_path,excluded_sites,freq_min=300,freq_max=6000,MOCAP_200Hz_notch=True,remove_stim_artefact=True,Plotting=True):
    #Check if concatenated file already exists
    if os.path.isdir(rf'{saving_dir}/{saving_name}/'):
        print('Concatenated file already exists')
        rec_binary = si.load_extractor(rf'{saving_dir}/{saving_name}/')
    
    
    
    else:
        print('Concatenating...')
        
        
        """------------------Concatenation------------------"""
        recordings_list=[]
        for recording_file in recordings:
            recording = se.read_intan(recording_file,stream_id='0')
            recording.annotate(is_filtered=False)
            recordings_list.append(recording)
        
        multirecording = si.concatenate_recordings(recordings_list)
        
        # recording_info = Get_recordings_info(saving_name,saving_dir,recordings,freq_min,freq_max,probe_path,excluded_sites)
        recording_info_path = os.path.dirname((os.path.dirname(recordings[0])))
        recording_info = pickle.load(open(rf'{recording_info_path}/ttl_idx.pickle', "rb"))

                 

        """------------------Set the probe------------------"""
        probe = pi.io.read_probeinterface(probe_path)
        probe = probe.probes[0]
        multirecording = multirecording.set_probe(probe)
        if Plotting==True:
            plot_probe(probe, with_channel_index=True, with_device_index=True)

        
        """------------------Defective sites exclusion------------------"""
        if Plotting==True:
            sw.plot_timeseries(multirecording, channel_ids=multirecording.get_channel_ids(),time_range=[10,30])
        
        multirecording.set_channel_groups(1, excluded_sites)
        multirecording = multirecording.split_by('group')[0]
        if Plotting==True:
            sw.plot_timeseries(multirecording, channel_ids=multirecording.get_channel_ids(),time_range=[10,30])
        
        
        """------------------Pre Processing------------------"""
        #Bandpass filter
        recording_f = spre.bandpass_filter(multirecording, freq_min=freq_min, freq_max=freq_max)
        if Plotting==True:
            w = sw.plot_timeseries(recording_f,time_range=[10,30], segment_index=0)
        
        
        if MOCAP_200Hz_notch == True:
            for i in [num for num in range(300, 6000 + 1) if num % 200 == 0]:
                recording_f = spre.notch_filter(recording_f, freq=i)
        
        
        if remove_stim_artefact == True:
            
            
            stim_idx = recording_info['stim_ttl_on']
            
            
            recording_f = spre.remove_artifacts(recording_f,stim_idx, ms_before=1, ms_after=1)
            
            
        
        
        #Median common ref
        recording_cmr = spre.common_reference(recording_f, reference='global', operator='median')
        if Plotting==True:
            w = sw.plot_timeseries(recording_cmr,time_range=[10,30], segment_index=0)
        
        rec_binary = recording_cmr.save(folder=rf'{saving_dir}/{saving_name}/', n_jobs=1, progress_bar=True, chunk_duration='1s')
     
       
        trial_time_index_df=pd.DataFrame({'concatenated_time':multirecording.get_times()})


        with open(rf'{saving_dir}/{saving_name}/concatenated_recording_trial_time_index_df.pickle', 'wb') as file:
            pickle.dump(trial_time_index_df, file, protocol=pickle.HIGHEST_PROTOCOL)   
            
        # pickle.dump(recording_info, open(rf"{saving_dir}/{saving_name}/recordings_info.pickle", "wb"))

    return rec_binary

    
    

def spike_sorting(record,spikesorting_results_folder,saving_name,use_docker=True,nb_of_agreement=2,plot_sorter=True,plot_comp=True,save=True):
    """
    Perform spike sorting using multiple sorters and compare the results.

    Parameters:
        record (spikeinterface.RecordingExtractor): The recording extractor to be sorted.
        spikesorting_results_folder (str): Directory where the results of spike sorting will be saved.
        saving_name (str): Name used for creating subdirectories for each sorter's results.
        use_docker (bool, optional): Whether to use Docker to run the spike sorters. Defaults to True.
        nb_of_agreement (int, optional): Minimum number of sorters that must agree to consider a unit as valid. Defaults to 2.
        plot_sorter (bool, optional): Whether to enable plotting of individual sorter's results. Defaults to True.
        plot_comp (bool, optional): Whether to enable plotting of comparison results. Defaults to True.
        save (bool, optional): Whether to save the plot images. Defaults to True.

    Returns:
        None
    """
    
    param_sorter = {
        'kilosort3':{},   
        'mountainsort4': {
                                        'detect_sign': -1,  # Use -1, 0, or 1, depending on the sign of the spikes in the recording
                                        'adjacency_radius': -1,  # Use -1 to include all channels in every neighborhood
                                        'freq_min': 300,  # Use None for no bandpass filtering
                                        'freq_max': 6000,
                                        'filter': True,
                                        'whiten': True,  # Whether to do channel whitening as part of preprocessing
                                        'num_workers': 1,
                                        'clip_size': 50,
                                        'detect_threshold': 3,
                                        'detect_interval': 10,  # Minimum number of timepoints between events detected on the same channel
                                        'tempdir': None
                                    },
                    # 'tridesclous': {
                    #                     'freq_min': 300.,   #'High-pass filter cutoff frequency'
                    #                     'freq_max': 6000.,#'Low-pass filter cutoff frequency'
                    #                     'detect_sign': -1,     #'Use -1 (negative) or 1 (positive) depending on the sign of the spikes in the recording',
                    #                     'detect_threshold': 5, #'Threshold for spike detection',
                    #                     'n_jobs' : 8,           #'Number of jobs (when saving ti binary) - default -1 (all cores)',
                    #                     'common_ref_removal': True,     #'remove common reference with median',
                    #                 },
                    'spykingcircus': {
                                        'detect_sign': -1,  #'Use -1 (negative),1 (positive) or 0 (both) depending on the sign of the spikes in the recording'
                                        'adjacency_radius': 100,  # Radius in um to build channel neighborhood
                                        'detect_threshold': 6,  # Threshold for detection
                                        'template_width_ms': 3,  # Template width in ms. Recommended values: 3 for in vivo - 5 for in vitro
                                        'filter': True, # Enable or disable filter
                                        'merge_spikes': True, #Enable or disable automatic mergind
                                        'auto_merge': 0.75, #Automatic merging threshold
                                        'num_workers': None, #Number of workers (if None, half of the cpu number is used)
                                        'whitening_max_elts': 1000,  # Max number of events per electrode for whitening
                                        'clustering_max_elts': 10000,  # Max number of events per electrode for clustering
                                    }
                 }
    
    print("Spike sorting starting")

    sorter_list = []
    sorter_name_list = []
    for sorter_name, sorter_param in param_sorter.items():
        print(sorter_name)
        
        output_folder = rf'{spikesorting_results_folder}\{saving_name}\{sorter_name}'
        
        if os.path.isdir(output_folder):
            print('Sorter folder found, load from folder')
            sorter_result = ss.NpzSortingExtractor.load_from_folder(rf'{output_folder}/in_container_sorting')
        else:
            sorter_result = ss.run_sorter(sorter_name,recording=record,output_folder=output_folder,docker_image=True,verbose=False,**sorter_param)
        
              
        sorter_list.append(sorter_result)
        sorter_name_list.append(sorter_name)
    

        #save the sorter params
        with open(f'{output_folder}\param_dict.pkl', 'wb') as f:
            pickle.dump(sorter_param, f)
        if os.path.isdir(f'{output_folder}\we'):
            print('Waveform folder found, load from folder')
            we = si.WaveformExtractor.load_from_folder(f'{output_folder}\we', sorting=sorter_result)
        else:
            we = si.extract_waveforms(record, sorter_result, folder=f'{output_folder}\we')
    
        if plot_sorter:
            print('Plot sorting summary in progress')
            plot_maker(sorter_result, we, save, sorter_name, spikesorting_results_folder,saving_name)
            print('Plot sorting summary finished')
        print('================================')
    
    print("Spike sorting done")
    
    
    
    if len(sorter_list) > 1 and nb_of_agreement != 0:
        ############################
        # Sorter outup comparaison #
        base_comp_folder = rf'{spikesorting_results_folder}\{saving_name}'
        comp_multi_name = f'comp_mult_{nb_of_agreement}'
        
        for sorter_name in sorter_name_list:
            comp_multi_name += f'_{sorter_name}'
        base_comp_folder = f'{base_comp_folder}\{comp_multi_name}'

        if os.path.isdir(f'{base_comp_folder}\sorter'):
            print('multiple comparaison sorter folder found, load from folder')
            sorting_agreement = ss.NpzSortingExtractor.load_from_folder(f'{base_comp_folder}\sorter')
        else:
            print('multiple comparaison sorter folder not found, computing from sorter list')
            comp_multi = sc.compare_multiple_sorters(sorting_list=sorter_list,
                                                    name_list=sorter_name_list)
            comp_multi.save_to_folder(base_comp_folder)
            # del sorting_list, sorting_name_list
            sorting_agreement = comp_multi.get_agreement_sorting(minimum_agreement_count=nb_of_agreement)
            sorting_agreement._is_json_serializable = False

            sorting_agreement.save_to_folder(f'{base_comp_folder}\sorter')
        
        
        try:
            we = si.extract_waveforms(record, sorting_agreement, folder=f'{base_comp_folder}\we')
        
        
        except FileExistsError:
            print('multiple comparaison waveform folder found, load from folder')
            we = si.WaveformExtractor.load_from_folder(f'{base_comp_folder}\we', sorting=sorting_agreement)
        
        
        if plot_comp:
            print('Plot multiple comparaison summary in progress')
            plot_maker(sorting_agreement, we, save, comp_multi_name, spikesorting_results_folder,saving_name)
            print('Plot multiple comparaison summary finished\n')

    return

def plot_maker(sorter, we, save, sorter_name, save_path,saving_name):
    """
    Generate and save plots for an individual sorter's results.
    
    Parameters:
        sorter (spikeinterface.SortingExtractor): The sorting extractor containing the results of a spike sorter.
        we (spikeinterface.WaveformExtractor): The waveform extractor for the sorting extractor.
        save (bool): Whether to save the generated plots.
        sorter_name (str): Name of the spike sorter.
        save_path (str): Directory where the plots will be saved.
        saving_name (str): Name of the recording data.
        
    Returns:
        None
    """
    
    for unit_id in sorter.get_unit_ids():
        fig = plt.figure(figsize=(25, 13))
        gs = GridSpec(nrows=3, ncols=6)
        fig.suptitle(f'{sorter_name}\n{saving_name}\nunits {unit_id} (Total spike {sorter.get_total_num_spikes()[unit_id]})',)
        ax0 = fig.add_subplot(gs[0, 0:3])
        ax1 = fig.add_subplot(gs[0, 3:7])
        ax1.set_title('Mean firing rate during a trial')
        ax2 = fig.add_subplot(gs[1, :])
        ax2.set_title('Waveform of the unit')
        ax3 = fig.add_subplot(gs[2, 0])
        ax4 = fig.add_subplot(gs[2, 1], sharey = ax3)
        ax5 = fig.add_subplot(gs[2, 2], sharey = ax3)
        ax6 = fig.add_subplot(gs[2, 3:6])
        sw.plot_autocorrelograms(sorter, unit_ids=[unit_id], axes=ax0, bin_ms=1, window_ms=200)
        ax0.set_title('Autocorrelogram')
        current_spike_train = sorter.get_unit_spike_train(unit_id)/sorter.get_sampling_frequency()
        current_spike_train_list = []
        while len(current_spike_train) > 0: #this loop is to split the spike train into trials with correct duration in seconds
            # Find indices of elements under 9 (9 sec being the duration of the trial)
            indices = np.where(current_spike_train < 9)[0]
            if len(indices)>0:
                # Append elements to the result list
                current_spike_train_list.append(SpikeTrain(current_spike_train[indices]*s, t_stop=9))
                # Remove the appended elements from the array
                current_spike_train = np.delete(current_spike_train, indices)
                # Subtract 9 from all remaining elements
            current_spike_train -= 9
        bin_size = 100
        histogram = time_histogram(current_spike_train_list, bin_size=bin_size*ms, output='mean')
        histogram = histogram*(1000/bin_size)
        ax1.axvspan(0, 0.5, color='green', alpha=0.3)
        ax1.axvspan(1.5, 2, color='green', alpha=0.3)
        ax6.axvspan(0, 0.5, color='green', alpha=0.3)
        ax6.axvspan(1.5, 2, color='green', alpha=0.3)
        plot_time_histogram(histogram, units='s', axes=ax1)
        sw.plot_unit_waveforms_density_map(we, unit_ids=[unit_id], ax=ax2)
        template = we.get_template(unit_id=unit_id).copy()
        
        for curent_ax in [ax3, ax4, ax5]:
            max_channel = np.argmax(np.abs(template))%template.shape[1]
            template[:,max_channel] = 0
            mean_residual = np.mean(np.abs((we.get_waveforms(unit_id=unit_id)[:,:,max_channel] - we.get_template(unit_id=unit_id)[:,max_channel])), axis=0)
            curent_ax.plot(mean_residual)
            curent_ax.plot(we.get_template(unit_id=unit_id)[:,max_channel])
            curent_ax.set_title('Mean residual of the waveform for channel '+str(max_channel))
        plt.tight_layout()
        rasterplot_rates(current_spike_train_list, ax=ax6, histscale=0.1)
        if save:
            plt.savefig(fr'{save_path}\{saving_name}\{sorter_name}\we\Unit_{int(unit_id)}.pdf')
            plt.savefig(fr'{save_path}\{saving_name}\{sorter_name}\we\Unit_{int(unit_id)}.png')
            plt.close()


#%%Parameters

#####################################################################
###################### TO CHANGE ####################################
#####################################################################
#Folder containing the folders of the session
rhd_folder = r'D:\ePhy\Intan_Data\0026\0026_29_07'
saving_name="0026_29_07"



#####################################################################
#Verify the following parameters and paths

probe_path=r'D:/ePhy/SI_Data/A1x16-Poly2-5mm-50s-177.json'   #INTAN Optrode
# probe_path = 'D:/ePhy/SI_Data/Buzsaki16.json'              #INTAN Buzsaki16


# Saving Folder path
saving_dir=r"D:/ePhy/SI_Data/concatenated_signals"
spikesorting_results_folder='D:\ePhy\SI_Data\spikesorting_results'


# Sites to exclude
excluded_sites = []


#%%
recordings = list_recording_files(rhd_folder)

    
recording = concatenate_preprocessing(recordings,saving_dir,saving_name,
                                      probe_path,excluded_sites,Plotting=True,
                                      freq_min=300, freq_max=6000,
                                      MOCAP_200Hz_notch=True,
                                      remove_stim_artefact=True)


# spike_sorting(recording,spikesorting_results_folder,saving_name,plot_sorter=True, plot_comp=True)