# -*- coding: utf-8 -*-
#! /bin/env python
"""
Created on Tue Feb 28 16:06:17 2023

@author: Gilles.DELBECQ
"""

import sys, struct, math, os, time
import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as sp

from intanutil.read_header import read_header
from intanutil.get_bytes_per_data_block import get_bytes_per_data_block
from intanutil.read_one_data_block import read_one_data_block
from intanutil.notch_filter import notch_filter
from intanutil.data_to_result import data_to_result

import pandas as pd


def read_data(filename):
    """Reads Intan Technologies RHD2000 data file generated by evaluation board GUI.
    
    Data are returned in a dictionary, for future extensibility.
    """
    from intanutil.read_header import read_header
    from intanutil.get_bytes_per_data_block import get_bytes_per_data_block
    from intanutil.read_one_data_block import read_one_data_block
    from intanutil.notch_filter import notch_filter
    from intanutil.data_to_result import data_to_result   
    

    tic = time.time()
    fid = open(filename, 'rb')
    filesize = os.path.getsize(filename)

    header = read_header(fid)

    print('Found {} amplifier channel{}.'.format(header['num_amplifier_channels'], plural(header['num_amplifier_channels'])))
    print('Found {} auxiliary input channel{}.'.format(header['num_aux_input_channels'], plural(header['num_aux_input_channels'])))
    print('Found {} supply voltage channel{}.'.format(header['num_supply_voltage_channels'], plural(header['num_supply_voltage_channels'])))
    print('Found {} board ADC channel{}.'.format(header['num_board_adc_channels'], plural(header['num_board_adc_channels'])))
    print('Found {} board digital input channel{}.'.format(header['num_board_dig_in_channels'], plural(header['num_board_dig_in_channels'])))
    print('Found {} board digital output channel{}.'.format(header['num_board_dig_out_channels'], plural(header['num_board_dig_out_channels'])))
    print('Found {} temperature sensors channel{}.'.format(header['num_temp_sensor_channels'], plural(header['num_temp_sensor_channels'])))
    print('')

    # Determine how many samples the data file contains.
    bytes_per_block = get_bytes_per_data_block(header)

    # How many data blocks remain in this file?
    data_present = False
    bytes_remaining = filesize - fid.tell()
    if bytes_remaining > 0:
        data_present = True

    if bytes_remaining % bytes_per_block != 0:
        raise Exception('Something is wrong with file size : should have a whole number of data blocks')

    num_data_blocks = int(bytes_remaining / bytes_per_block)

    num_amplifier_samples = header['num_samples_per_data_block'] * num_data_blocks
    num_aux_input_samples = int((header['num_samples_per_data_block'] / 4) * num_data_blocks)
    num_supply_voltage_samples = 1 * num_data_blocks
    num_board_adc_samples = header['num_samples_per_data_block'] * num_data_blocks
    num_board_dig_in_samples = header['num_samples_per_data_block'] * num_data_blocks
    num_board_dig_out_samples = header['num_samples_per_data_block'] * num_data_blocks

    record_time = num_amplifier_samples / header['sample_rate']

    if data_present:
        print('File contains {:0.3f} seconds of data.  Amplifiers were sampled at {:0.2f} kS/s.'.format(record_time, header['sample_rate'] / 1000))
    else:
        print('Header file contains no data.  Amplifiers were sampled at {:0.2f} kS/s.'.format(header['sample_rate'] / 1000))

    if data_present:
        # Pre-allocate memory for data.
        print('')
        print('Allocating memory for data...')

        data = {}
        if (header['version']['major'] == 1 and header['version']['minor'] >= 2) or (header['version']['major'] > 1):
            data['t_amplifier'] = np.zeros(num_amplifier_samples, dtype=np.int_)
        else:
            data['t_amplifier'] = np.zeros(num_amplifier_samples, dtype=np.uint)

        data['amplifier_data'] = np.zeros([header['num_amplifier_channels'], num_amplifier_samples], dtype=np.uint)
        data['aux_input_data'] = np.zeros([header['num_aux_input_channels'], num_aux_input_samples], dtype=np.uint)
        data['supply_voltage_data'] = np.zeros([header['num_supply_voltage_channels'], num_supply_voltage_samples], dtype=np.uint)
        data['temp_sensor_data'] = np.zeros([header['num_temp_sensor_channels'], num_supply_voltage_samples], dtype=np.uint)
        data['board_adc_data'] = np.zeros([header['num_board_adc_channels'], num_board_adc_samples], dtype=np.uint)
        
        # by default, this script interprets digital events (digital inputs and outputs) as booleans
        # if unsigned int values are preferred(0 for False, 1 for True), replace the 'dtype=np.bool_' argument with 'dtype=np.uint' as shown
        # the commented line below illustrates this for digital input data; the same can be done for digital out
        
        #data['board_dig_in_data'] = np.zeros([header['num_board_dig_in_channels'], num_board_dig_in_samples], dtype=np.uint)
        data['board_dig_in_data'] = np.zeros([header['num_board_dig_in_channels'], num_board_dig_in_samples], dtype=np.bool_)
        data['board_dig_in_raw'] = np.zeros(num_board_dig_in_samples, dtype=np.uint)
        
        data['board_dig_out_data'] = np.zeros([header['num_board_dig_out_channels'], num_board_dig_out_samples], dtype=np.bool_)
        data['board_dig_out_raw'] = np.zeros(num_board_dig_out_samples, dtype=np.uint)

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
            indices['aux_input'] += int(header['num_samples_per_data_block'] / 4)
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
        if bytes_remaining != 0: raise Exception('Error: End of file not reached.')



    # Close data file.
    fid.close()

    if (data_present):
        print('Parsing data...')

        # Extract digital input channels to separate variables.
        for i in range(header['num_board_dig_in_channels']):
            data['board_dig_in_data'][i, :] = np.not_equal(np.bitwise_and(data['board_dig_in_raw'], (1 << header['board_dig_in_channels'][i]['native_order'])), 0)

        # Extract digital output channels to separate variables.
        for i in range(header['num_board_dig_out_channels']):
            data['board_dig_out_data'][i, :] = np.not_equal(np.bitwise_and(data['board_dig_out_raw'], (1 << header['board_dig_out_channels'][i]['native_order'])), 0)

        # Scale voltage levels appropriately.
        data['amplifier_data'] = np.multiply(0.195, (data['amplifier_data'].astype(np.int32) - 32768))      # units = microvolts
        data['aux_input_data'] = np.multiply(37.4e-6, data['aux_input_data'])               # units = volts
        data['supply_voltage_data'] = np.multiply(74.8e-6, data['supply_voltage_data'])     # units = volts
        if header['eval_board_mode'] == 1:
            data['board_adc_data'] = np.multiply(152.59e-6, (data['board_adc_data'].astype(np.int32) - 32768)) # units = volts
        elif header['eval_board_mode'] == 13:
            data['board_adc_data'] = np.multiply(312.5e-6, (data['board_adc_data'].astype(np.int32) - 32768)) # units = volts
        else:
            data['board_adc_data'] = np.multiply(50.354e-6, data['board_adc_data'])           # units = volts
        data['temp_sensor_data'] = np.multiply(0.01, data['temp_sensor_data'])               # units = deg C

        # Check for gaps in timestamps.
        num_gaps = np.sum(np.not_equal(data['t_amplifier'][1:]-data['t_amplifier'][:-1], 1))
        if num_gaps == 0:
            print('No missing timestamps in data.')
        else:
            print('Warning: {0} gaps in timestamp data found.  Time scale will not be uniform!'.format(num_gaps))

        # Scale time steps (units = seconds).
        data['t_amplifier'] = data['t_amplifier'] / header['sample_rate']
        data['t_aux_input'] = data['t_amplifier'][range(0, len(data['t_amplifier']), 4)]
        data['t_supply_voltage'] = data['t_amplifier'][range(0, len(data['t_amplifier']), header['num_samples_per_data_block'])]
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
                data['amplifier_data'][i,:] = notch_filter(data['amplifier_data'][i,:], header['sample_rate'], header['notch_filter_frequency'], 10)

                fraction_done = 100 * (i / header['num_amplifier_channels'])
                if fraction_done >= percent_done:
                    print('{}% done...'.format(percent_done))
                    percent_done += print_increment
    else:
        data = [];

    # Move variables to result struct.
    result = data_to_result(header, data, data_present)

    print('Done!  Elapsed time: {0:0.1f} seconds'.format(time.time() - tic))
    return result

def plural(n):
    """Utility function to optionally pluralize words based on the value of n.
    """

    if n == 1:
        return ''
    else:
        return 's'


def filter_signal(signal, order=4, sample_rate=20000, freq_low=300, freq_high=3000, axis=0):
    """
    From Théo G.
    Filtering with scipy
    
    inputs raw signal (array)
    returns filtered signal (array)
    """
    
    import scipy.signal
    Wn = [freq_low / (sample_rate / 2), freq_high / (sample_rate / 2)]
    sos_coeff = scipy.signal.iirfilter(order, Wn, btype="band", ftype="butter", output="sos")
    filtered_signal = scipy.signal.sosfiltfilt(sos_coeff, signal, axis=axis)
    return filtered_signal



"""
Load intan file
"""
path=r"//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Février2023/Test_Gustave/raw/raw intan/Test_Gustave_16_03_230316_165717/merged.rhd"
savepath=r'\\equipe2-nas1\Gilles.DELBECQ\Data\ePhy\Février2023\Test_Gustave\figures_optotag'
save_prefix='16_03_optotag_5ms_2Hz'


reader=read_data(path)
sampling_rate = reader['frequency_parameters']['amplifier_sample_rate']
time_vector=reader['t_amplifier']
signal=reader['amplifier_data']
dig_inputs=reader['board_dig_in_data'][1]



"""
Récupérer l'index des stims'
"""
stim_idx=[]
for i in range(len(dig_inputs)):
    if dig_inputs[i] and (i==0 or not dig_inputs[i-1]):
        stim_idx.append(i)



"""
Filtering
"""
filtered_signals=[]

for i in range(len(signal)):
    filtered_signal=filter_signal(signal[int(i),:])
    filtered_signals.append(filtered_signal)
    # plt.figure()
    # # plt.plot(time_vector,signal[0,:])
    # plt.plot(time_vector,filtered_signal)
    # plt.title(rf'channel {int(i)}')

filtered_signals = np.array(filtered_signals)
median = np.median(filtered_signals, axis=0)#compute median on all 
cmr_signals = filtered_signals-median     #compute common ref removal median on all 



"""
Spike detection
"""
# Noise parameters
std_threshold = 4 #Times the std
noise_window = 2 #window for the noise calculation in sec
distance = 50 # distance between 2 spikes

waveform_window=10#⌡in ms

thresholds,spikes_list,spikes_list_y,spikes_times_list,wfs,waveforms=[],[],[],[],[],[]

for signal in cmr_signals:
    # Threshold calculation
    noise = signal[0:int(noise_window*sampling_rate)] #noise window taken from individual channel signal
    threshold = np.median(noise)+std_threshold*np.std(noise) #threshold calculation for the channel
    thresholds.append(threshold) #append it to the list regrouping threshold for each channel
    
    
    #Detect the spike indexes
    spike_idx, _ = sp.find_peaks(-signal,height=threshold,distance=distance)
    #Convert to spike times
    spike_times = spike_idx*1./sampling_rate
    #Get spikes peak 
    spike_y = signal[spike_idx]
    
    #Append spikes times to the list of all channels spikes
    spikes_list.append(spike_idx)
    spikes_times_list.append(spike_times)
    spikes_list_y.append(spike_y)
    
    # if Waveforms == True :
    #     wfs = extract_spike_waveform(signal,spike_idx)
    #     waveforms.append(wfs)

for index,i in np.ndenumerate(waveforms):
    plt.figure()
    # plt.title(rf'waveform_chan_{selected_chan[index[0]]}')
    time_axis=np.array(range(int(-(waveform_window/1000)*20000/2),int(waveform_window/1000*20000/2)))/20000*1000
    for j in i:
        plt.plot(j*1000)
    # plt.savefig(rf'{save_path}\waveform_chan_{selected_chan[index[0]]}.svg')




"""
Raster plot

"""
for Channel_to_analyze in range(16):
    
    
    event_indices=spikes_list[Channel_to_analyze]
    signal=cmr_signals[Channel_to_analyze]
    
    
    # Durée de la fenêtre avant et après chaque stimulation (en nombre de points)
    pre_stim_window_size = int(10*sampling_rate/1000)
    post_stim_window_size = int(50*sampling_rate/1000)
    
    # Durée de chaque fenêtre (en nombre de points)
    window_size = int(pre_stim_window_size + post_stim_window_size)
    
    # Création du tableau 2D pour stocker les fenêtres de signal
    num_stimulations = len(stim_idx)
    num_events = len(event_indices)
    signal_windows = np.zeros((num_events, num_stimulations, window_size))
    
    num_windows = num_stimulations
    
    event_times = spikes_times_list
    stim_times = np.array(stim_idx)/sampling_rate*1000
    
    plt.figure()
    plt.plot(time_vector[500:100000],signal[500:100000])

    # for spike in event_indices/sampling_rate:
    #     plt.axvline(spike,color='red')
    
    
    """
    #Raster plot global
    plt.figure()
    plt.eventplot(event_times, colors='k')
    """
    
    
    # Indices des pics
    peak_indices = spikes_list[Channel_to_analyze]
    
    # Nombre total de fenêtres
    num_windows = len(stim_idx)
    
    # Créer une matrice de zéros pour stocker les fenêtres de signal
    signal_windows = np.zeros((num_windows, pre_stim_window_size + post_stim_window_size))
    
    # Remplir la matrice avec les fenêtres de signal
    for i, stim_index in enumerate(stim_idx):
        signal_windows[i, :] = signal[stim_index - pre_stim_window_size : stim_index + post_stim_window_size]
    
    # Créer une matrice de zéros pour stocker les événements
    event_matrix = np.zeros_like(signal_windows)
    
    # Trouver les indices des pics dans chaque fenêtre de signal
    for i in range(num_windows):
        peak_indices_in_window = peak_indices[(peak_indices >= stim_idx[i] - pre_stim_window_size) & (peak_indices < stim_idx[i] + post_stim_window_size)]
        if len(peak_indices_in_window) >0:
            event_matrix[i, peak_indices_in_window - (stim_idx[i] - pre_stim_window_size)] = 1
    
    
    
    
    from itertools import groupby
    
    # votre array
    arr = np.argwhere(event_matrix == 1)
    
    # Regrouper les valeurs de la deuxième colonne par les valeurs de la première colonne
    grouped_array = []
    for i in np.unique(arr[:, 0]):
        grouped_array.append(list(arr[arr[:, 0] == i, 1]))
    
    # Créer un nouveau 2D array avec les valeurs de la deuxième colonne regroupées
    new_array = np.array([np.array(x) for x in grouped_array])
    
    
    
    
    
    # Afficher le raster plot
    plt.figure()
    plt.eventplot(np.array(new_array), colors='k')
    
    # Ajouter une ligne verticale pour chaque stimulation
    # plt.axvline(200, color='r', linestyle='--')
    plt.axvspan(200,300,color='blue',alpha=0.2)
    
    # Ajouter une étiquette pour l'axe des y
    plt.ylabel('Stimulation')
    plt.title(rf'Channel : {Channel_to_analyze}')

    plt.savefig(rf"{savepath}/{save_prefix}_Channel_{Channel_to_analyze}.png")
    
    
    test= pd.DataFrame(grouped_array)
    test.to_excel(rf"{savepath}/{save_prefix}_Channel_{Channel_to_analyze}.xlsx")
    
    
    num_events = int(np.sum(event_matrix))
    event_times = np.where(event_matrix == 1)
    
    # events_to_plot=event_times[1]
    
    events_to_plot=np.where((event_times[1] >= 195) & (event_times[1] <= 205), np.nan, event_times[1])
    events_to_plot=np.where((events_to_plot >= 295) & (events_to_plot <= 305), np.nan, events_to_plot)
    
    
    bins_in_ms=0.5
    plt.figure()
    plt.hist(events_to_plot, bins=int(1200/bins_in_ms/sampling_rate*1000))
    plt.axvspan(200,300,color='blue',alpha=0.2)
    plt.xlim(0,500)
    plt.title(rf'Channel {Channel_to_analyze} bins = {bins_in_ms}ms')
    plt.savefig(rf"{savepath}/{save_prefix}_Hist_Channel_{Channel_to_analyze}_no_artefact.svg")
    plt.savefig(rf"{savepath}/{save_prefix}_Hist_Channel_{Channel_to_analyze}_no_artefact.png")
    
    