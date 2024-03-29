# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:36:00 2023

@author: MOCAP
"""


#%% Functions and imports

import spikeinterface as si


import sys, struct, math, os, time
import numpy as np

sys.path.append(r"C:\Users\MOCAP\Documents\GitHub\main\In vivo Multi")

from intanutil.read_header import read_header
from intanutil.get_bytes_per_data_block import get_bytes_per_data_block
from intanutil.read_one_data_block import read_one_data_block
from intanutil.notch_filter import notch_filter
from intanutil.data_to_result import data_to_result

import matplotlib.pyplot as plt


def read_data(filename):
    """Reads Intan Technologies RHD2000 data file generated by evaluation board GUI.
    
    Data are returned in a dictionary, for future extensibility.
    """

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


def filter_signal(signal, order=3, sample_rate=20000, freq_low=300, freq_high=6000, axis=0):
    import scipy.signal
    Wn = [freq_low / (sample_rate / 2), freq_high / (sample_rate / 2)]
    sos_coeff = scipy.signal.iirfilter(order, Wn, btype="band", ftype="butter", output="sos")
    filtered_signal = scipy.signal.sosfiltfilt(sos_coeff, signal, axis=axis)
    return filtered_signal


def extract_spike_waveform(signal, spike_idx, left_width=(5/1000)*20000/2, right_width=(5/1000)*20000/2):
    
    '''
    Function to extract spikes waveforms in spike2 recordings
    
    INPUTS :
        signal (1-d array) : the ephy signal
        spike_idx (1-d array or integer list) : array containing the spike indexes (in points)
        width (int) = width for spike window
    
    OUTPUTS : 
        SPIKES (list) : a list containg the waveform of each spike 
    
    '''
    
    SPIKES = []
    
    left_width = int(left_width)
    right_width = int(right_width)
    
    for i in range(len(spike_idx)): 
        index = spike_idx[i]

        spike_wf = signal[index-left_width : index+right_width]

        SPIKES.append(spike_wf)
    return SPIKES


def get_file_names(directory):
    file_names = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_names.append(file_path)
    return file_names


#%% files loading

spike_times = np.load('C:/Users/MOCAP/Desktop/temp/0012_24_05_all_01_06/phy_export/tridesclous/spike_times.npy')
spike_cluster = np.load('C:/Users/MOCAP/Desktop/temp/0012_24_05_all_01_06/phy_export/tridesclous/spike_clusters.npy')
spike_templates = np.load('C:/Users/MOCAP/Desktop/temp/0012_24_05_all_01_06/phy_export/tridesclous/similar_templates.npy')

recordings = [  
"D:/ePhy/Intan_Data/0012/05_24/0014_24_05_230524_163229/0014_24_05_230524_163229.rhd",
"D:/ePhy/Intan_Data/0012/05_24/0014_24_05_230524_163605/0014_24_05_230524_163605.rhd",
"D:/ePhy/Intan_Data/0012/05_24/0014_24_05_230524_163928/0014_24_05_230524_163928.rhd",
"D:/ePhy/Intan_Data/0012/05_24/0014_24_05_230524_164414/0014_24_05_230524_164414.rhd"]


savefig_folder=r"C:\Users\MOCAP\Desktop\Spikesorting_06_06"


multi_recordings,recordings_lengths,multi_stim_idx=[],[],[]

#Concatenate recordings
for record in recordings:
    reader=read_data(record)
    signal = reader['amplifier_data'] 
    recordings_lengths.append(len(signal[0]))
    multi_recordings.append(signal)  
    
    stim_idx=reader['board_dig_in_data'][0]
    multi_stim_idx.append(stim_idx)

concatenated_signal = np.hstack(multi_recordings)
concatenated_stim_idx=np.hstack(multi_stim_idx)


#Get sampling freq
frequency=reader['frequency_parameters']['amplifier_sample_rate']

#Get spikes for each cluster 
clusters_idx = np.unique(spike_cluster)
selected_spike_times,selected_spike_indexes=[],[]

for cluster in clusters_idx:
    array_idx = np.where(spike_cluster==cluster)[0]
    selected_spike_idx = np.take(spike_times,array_idx)
    selected_spike_indexes.append(selected_spike_idx)
    selected_spike_times.append(selected_spike_idx/frequency) #in seconds


#Get digital inputs
# stim_idx=reader['board_dig_in_data'][0]
# frame_idx=reader['board_dig_in_data'][1]




sites_positions=[[0.0, 250.0],
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

channel_order=[12, 13, 14, 15, 11, 10, 9, 8, 7, 6, 5, 4, 0, 1, 2, 3]

channel_positions=list(zip(channel_order,sites_positions))




#%% Filtering preprocessing
#Filter signal for each channel
selected_chan=[0,1,2,3,4,5,8,12,13]

filtered_signals=[]

for i in range(len(concatenated_signal)):
    if i in selected_chan:
        signal_filtered = filter_signal(concatenated_signal[i]) #Filter the signal
        filtered_signals.append(signal_filtered) #Append it in list
        
filtered_signals = np.array(filtered_signals) #Transform it in array

# Calculate the median signal from all filtered signals
median = np.mean(filtered_signals, axis=0)

# Calculate the cmr signals for each channels
cmr_signals = filtered_signals-median     




#%% raster
   
lineoffsets1 = np.array([0.1, 0.1,0.1,0.1,0.1,0.1,0.1])
linelengths1 = [0.1, 0.1,0.1,0.1,0.1,0.1,0.1]

colors1 = ['C{}'.format(i) for i in range(len(selected_spike_times))]
plt.eventplot(selected_spike_times,colors=colors1, linelengths=1, lineoffsets=1, linewidths=1)

number_spike_cum = np.arange(len(selected_spike_times[3])) + 1
plt.plot(selected_spike_times[3], number_spike_cum, marker='o')


#%% waveforms
for cluster in clusters_idx:
    spike_idx=selected_spike_indexes[cluster]
    all_mean_wvfs=[]
    for channel in range(len(selected_chan)):
        wvfs = extract_spike_waveform(cmr_signals[channel],spike_idx)
        mean_wvfs = np.mean(wvfs, axis=0)
        all_mean_wvfs.append(mean_wvfs)
        
    for chan in range(len(all_mean_wvfs)):
        plt.figure()
        plt.plot(all_mean_wvfs[chan])
        plt.title(rf'Cluster : {cluster} Channel :{selected_chan[chan]}')
 

selected_positions = [pos for pos in channel_positions if pos[0] in selected_chan]



waveform_positions = []
for i, t in enumerate(selected_positions):
    new_tuple = t + (all_mean_wvfs[i],)
    waveform_positions.append(new_tuple)





"""



import numpy as np
import matplotlib.pyplot as plt

# Liste de tuples contenant l'identifiant du canal, les coordonnées (x, y) et les waveforms moyennes
canal_data = [(1, (2, 3), np.array([0.1, 0.2, 0.3, 0.2, 0.1])),
              (2, (5, 6), np.array([-0.2, -0.1, 0.0, -0.1, -0.2])),
              (3, (8, 9), np.array([0.3, 0.4, 0.5, 0.4, 0.3]))]

# Tracer les positions des canaux
x_coords = [coord[0] for _, coord, _ in waveform_positions]
y_coords = [coord[1] for _, coord, _ in waveform_positions]
plt.scatter(x_coords, y_coords)

# Ajouter les identifiants des canaux à leurs positions respectives
for canal_id, coord, waveform in waveform_positions:
    plt.annotate(str(canal_id), xy=coord, xytext=(5, -5),
                 textcoords='offset points', ha='right', va='bottom')

    # Créer un nouvel axe pour la waveform
    waveform_ax = plt.axes([coord[0] + 0.2, coord[1] + 0.2, 0.3, 0.3])  # Définir la position et la taille de l'axe

    # Tracer la waveform moyenne
    waveform_ax.plot(waveform, color='r')

    # Configurer les limites de l'axe x de la waveform
    waveform_ax.set_xlim(0, len(waveform) - 1)

# Configurer les labels des axes et le titre
plt.xlabel('Coordonnée x')
plt.ylabel('Coordonnée y')
plt.title('Position des canaux avec les waveforms moyennes')

# Afficher la figure
plt.show()
"""

"""
# Liste de tuples contenant l'identifiant du canal, les coordonnées (x, y) et les waveforms moyennes
canal_data = [(1, (2, 3), np.array([0.1, 0.2, 0.3, 0.2, 0.1])),
              (2, (5, 6), np.array([-0.2, -0.1, 0.0, -0.1, -0.2])),
              (3, (8, 9), np.array([0.3, 0.4, 0.5, 0.4, 0.3]))]

# Tracer les positions des canaux
x_coords = [coord[0] for _, coord, _ in waveform_positions]
y_coords = [coord[1] for _, coord, _ in waveform_positions]
plt.scatter(x_coords, y_coords)

# Ajouter les identifiants des canaux à leurs positions respectives
for canal_id, coord, waveform in waveform_positions:
    plt.annotate(str(canal_id), xy=coord, xytext=(5, -5),
                 textcoords='offset points', ha='right', va='bottom')

    # Tracer la waveform moyenne à côté du canal
    waveform_x = coord[0] + 0.5  # Décalage horizontal
    waveform_y = coord[1]  # Même position verticale
    plt.plot(waveform_x + np.arange(len(waveform)), waveform + waveform_y, color='r')

    # Changer l'échelle en x des waveforms
    waveform_xlim = (waveform_x, waveform_x + len(waveform) - 1)
    plt.xlim(waveform_xlim)

# Masquer les axes pour les waveforms
plt.gca().get_xaxis().set_visible(False)
plt.gca().get_yaxis().set_visible(False)

# Configurer les labels des axes et le titre
plt.xlabel('Coordonnée x')
plt.ylabel('Coordonnée y')
plt.title('Position des canaux avec les waveforms moyennes')

# Afficher la figure
plt.show()

"""






    
"Select channels, or plot everything"

#%% Vidéo

# frame_ttl=(np.where(frame_idx == True))[0]

# # Calculer la différence entre les éléments consécutifs
# diff_indices = np.diff(frame_ttl)

# # Trouver les indices où la différence n'est pas égale à 1
# phase_indices = np.where(diff_indices != 1)[0]-1

# # Extraire le premier index de chaque phase
# screenshot_idexes = frame_ttl[phase_indices]

# video_files = get_file_names(r'D:\SOD_2023\0014_24_05')


# selected_spike_times.append()

# plt.eventplot(screenshot_idexes/frequency,linelengths=1, lineoffsets=1, linewidths=1)

#%%Test elephant
from neo.core import SpikeTrain

from quantities import ms, s, Hz
from elephant.spike_train_generation import homogeneous_poisson_process, homogeneous_gamma_process
from elephant.statistics import mean_firing_rate

from elephant.statistics import time_histogram, instantaneous_rate

spiketrain1 = SpikeTrain(selected_spike_times[0]*s, t_stop=700)#Artefact stim
spiketrain2 = SpikeTrain(selected_spike_times[1]*s, t_stop=700)#Artefact stim
spiketrain3 = SpikeTrain(selected_spike_times[2]*s, t_stop=700)
spiketrain4 = SpikeTrain(selected_spike_times[3]*s, t_stop=700)
spiketrain5 = SpikeTrain(selected_spike_times[4]*s, t_stop=700)


plt.figure(figsize=(8, 3))
plt.eventplot([spiketrain1.magnitude,spiketrain2.magnitude,spiketrain3.magnitude,spiketrain4.magnitude,spiketrain5.magnitude], linelengths=0.75, color='black')
plt.xlabel('Time (s)', fontsize=16)
plt.yticks([0,1,2,3,4], fontsize=16)
plt.title("Figure 1");

print("The mean firing rate of spiketrain1 is", mean_firing_rate(spiketrain1))
print("The mean firing rate of spiketrain2 is", mean_firing_rate(spiketrain2))
print("The mean firing rate of spiketrain3 is", mean_firing_rate(spiketrain3))
print("The mean firing rate of spiketrain4 is", mean_firing_rate(spiketrain4))
print("The mean firing rate of spiketrain5 is", mean_firing_rate(spiketrain5))

spiketrains_list=[spiketrain1,spiketrain2,spiketrain3,spiketrain4,spiketrain5]

for spiketrain_name,spiketrain in enumerate(spiketrains_list):
   
    histogram_count = time_histogram([spiketrain], 0.5*s)
    
    print(type(histogram_count), f"of shape {histogram_count.shape}: {histogram_count.shape[0]} samples, {histogram_count.shape[1]} channel")
    print('sampling rate:', histogram_count.sampling_rate)
    print('times:', histogram_count.times)
    print('counts:', histogram_count.T[0])
    
    histogram_rate = time_histogram([spiketrain],  0.5*s, output='rate')
    
    print('times:', histogram_rate.times)
    print('rate:', histogram_rate.T[0])
    
    
    
    inst_rate = instantaneous_rate(spiketrain, sampling_period=3*s)
    
    print(type(inst_rate), f"of shape {inst_rate.shape}: {inst_rate.shape[0]} samples, {inst_rate.shape[1]} channel")
    print('sampling rate:', inst_rate.sampling_rate)
    print('times (first 10 samples): ', inst_rate.times[:10])
    print('instantaneous rate (first 10 samples):', inst_rate.T[0, :10])
    
    
    
    
    from elephant.kernels import GaussianKernel
    instantaneous_rate(spiketrain, sampling_period=20*ms, kernel=GaussianKernel(200*ms))
    
    plt.figure(dpi=150)
    
    # plotting the original spiketrain
    # plt.plot(spiketrain, [0]*len(spiketrain), 'r', marker=2, ms=25, markeredgewidth=2, lw=0, label='poisson spike times')
    
    # mean firing rate
    plt.hlines(mean_firing_rate(spiketrain,t_stop=400*s), xmin=spiketrain.t_start, xmax=spiketrain.t_stop, linestyle='--', label='mean firing rate')
    
    # time histogram
    plt.bar(histogram_rate.times, histogram_rate.magnitude.flatten(), width=histogram_rate.sampling_period,
            align='edge', alpha=0.3, label='time histogram (rate)',color='black')
    
    # instantaneous rate
    plt.plot(inst_rate.times.rescale(s), inst_rate.rescale(histogram_rate.dimensionality).magnitude.flatten(), label='instantaneous rate')
    
    
    #Length of each recordings
    [plt.axvline(_x, linewidth=1, color='g') for _x in np.cumsum(np.array(recordings_lengths)/frequency)]
    
    #Behavior phases
    starts=np.array([23,52,79,109])
    starts2=np.array([15,72,103,133])+131.3152
    starts3=np.array([9,54,80,107])+281.0944
    
    starts=np.concatenate((starts, starts2, starts3))
    
    stops=np.array([29,59,85,114])
    stops2=np.array([26,77,110,140])+131.3152
    stops3=np.array([30,62,90,112])+281.0944
    
    stops=np.concatenate((stops, stops2, stops3))
    
    lifts=np.array([7,36,70,94])
    lifts2=np.array([7,36,84,118])+131.3152
    lifts3=np.array([4,34,68,91])+281.0944
    
    lifts=np.concatenate((lifts, lifts2, lifts3))
    
    downs=np.array([21,45,76,105])
    downs2=np.array([14,43,89,124])+131.3152
    downs3=np.array([8,41,76,96])+281.0944
    
    downs=np.concatenate((downs, downs2, downs3))
    
    for i in range(len(starts)):
        plt.axvspan(starts[i], stops[i], alpha=0.5, color='red')
    for i in range(len(lifts)):
        plt.axvspan(lifts[i], downs[i], alpha=0.5, color='grey')
    
    
    # axis labels and legend
    plt.xlabel('time [{}]'.format(spiketrain.times.dimensionality.latex))
    plt.ylabel('firing rate [{}]'.format(histogram_rate.dimensionality.latex))
    plt.xlim(spiketrain.t_start, spiketrain.t_stop)
    plt.legend()
    plt.title(rf'Spiketrain {spiketrain_name+1}')
    plt.show()
    
    # plt.savefig(rf"{savefig_folder}/spiketrain_{spiketrain_name+1}.svg")
    

#%%Slicing activity
#Behavior phases
starts=np.array([23,52,79,109])
starts2=np.array([15,72,103,133])+131.3152
starts3=np.array([9,54,80,107])+281.0944

starts=np.concatenate((starts, starts2, starts3))

stops=np.array([29,59,85,114])
stops2=np.array([26,77,110,140])+131.3152
stops3=np.array([30,62,90,112])+281.0944

stops=np.concatenate((stops, stops2, stops3))

activity_times=list(zip(starts,stops))

spiketrains_activity=[]

for cluster in selected_spike_times:
    spike_times=cluster
    
    # Sélectionner les temps de potentiel d'action dans les phases d'activité
    activity_spikes = []
    for debut, fin in activity_times:
        mask = np.logical_and(spike_times >= debut, spike_times <= fin)
        temps_phase = spike_times[mask]
        activity_spikes.extend(temps_phase)

    # Convertir la liste en array
    activity_spikes = np.array(activity_spikes)

    # Afficher les temps de potentiel d'action sélectionnés
    print(activity_spikes)
    
    spiketrains_activity.append(activity_spikes)
    
    
    

spiketrain_activity3 = SpikeTrain(spiketrains_activity[2]*s, t_stop=410)
spiketrain_activity4 = SpikeTrain(spiketrains_activity[3]*s, t_stop=410)
spiketrain_activity5 = SpikeTrain(spiketrains_activity[4]*s, t_stop=410)

print("The mean firing rate of spiketrain3 is", mean_firing_rate(spiketrain_activity3))
print("The mean firing rate of spiketrain4 is", mean_firing_rate(spiketrain_activity4))
print("The mean firing rate of spiketrain5 is", mean_firing_rate(spiketrain_activity5))





#%%Slicing wait

downs=np.array([21,45,76,105])
downs2=np.array([14,43,89,124])+131.3152
downs3=np.array([8,41,76,96])+281.0944

downs=np.concatenate((downs, downs2, downs3))


starts=np.array([23,52,79,109])
starts2=np.array([15,72,103,133])+131.3152
starts3=np.array([9,54,80,107])+281.0944

starts=np.concatenate((starts, starts2, starts3))


wait_times=list(zip(downs,starts))

spiketrains_wait=[]

for cluster in selected_spike_times:
    spike_times=cluster
    
    # Sélectionner les temps de potentiel d'action dans les phases d'activité
    wait_spikes = []
    for debut, fin in wait_times:
        mask = np.logical_and(spike_times >= debut, spike_times <= fin)
        temps_phase = spike_times[mask]
        wait_spikes.extend(temps_phase)

    # Convertir la liste en array
    wait_spikes = np.array(wait_spikes)

    # Afficher les temps de potentiel d'action sélectionnés
    print(wait_spikes)
    
    spiketrains_wait.append(wait_spikes)
    
    
    

spiketrain_activity3 = SpikeTrain(spiketrains_wait[2]*s, t_stop=410)
spiketrain_activity4 = SpikeTrain(spiketrains_wait[3]*s, t_stop=410)
spiketrain_activity5 = SpikeTrain(spiketrains_wait[4]*s, t_stop=410)

print("The mean firing rate of spiketrain3 is", mean_firing_rate(spiketrain_activity3))
print("The mean firing rate of spiketrain4 is", mean_firing_rate(spiketrain_activity4))
print("The mean firing rate of spiketrain5 is", mean_firing_rate(spiketrain_activity5))



#%% Lift activity

lifts=np.array([7,36,70,94])
lifts2=np.array([7,36,84,118])+131.3152
lifts3=np.array([4,34,68,91])+281.0944

lifts=np.concatenate((lifts, lifts2, lifts3))

downs=np.array([21,45,76,105])
downs2=np.array([14,43,89,124])+131.3152
downs3=np.array([8,41,76,96])+281.0944

downs=np.concatenate((downs, downs2, downs3))


lift_times=list(zip(lifts,downs))

spiketrains_lift=[]

for cluster in selected_spike_times:
    spike_times=cluster
    
    # Sélectionner les temps de potentiel d'action dans les phases d'activité
    lift_spikes = []
    for debut, fin in lift_times:
        mask = np.logical_and(spike_times >= debut, spike_times <= fin)
        temps_phase = spike_times[mask]
        lift_spikes.extend(temps_phase)

    # Convertir la liste en array
    lift_spikes = np.array(lift_spikes)

    # Afficher les temps de potentiel d'action sélectionnés
    print(lift_spikes)
    
    spiketrains_lift.append(lift_spikes)
    
    
    

spiketrain_activity3 = SpikeTrain(spiketrains_lift[2]*s, t_stop=410)
spiketrain_activity4 = SpikeTrain(spiketrains_lift[3]*s, t_stop=410)
spiketrain_activity5 = SpikeTrain(spiketrains_lift[4]*s, t_stop=410)

print("The mean firing rate of spiketrain3 is", mean_firing_rate(spiketrain_activity3))
print("The mean firing rate of spiketrain4 is", mean_firing_rate(spiketrain_activity4))
print("The mean firing rate of spiketrain5 is", mean_firing_rate(spiketrain_activity5))

#%%Histogram mean rate by phase

means_by_phase=[]
for i in range(len(spiketrains_activity)):
    mean_activity=mean_firing_rate(SpikeTrain(spiketrains_activity[i]*s, t_stop=410))
    mean_wait=mean_firing_rate(SpikeTrain(spiketrains_wait[i]*s, t_stop=410))
    mean_lift=mean_firing_rate(SpikeTrain(spiketrains_lift[i]*s, t_stop=410))
    
    means_by_phase.append((mean_activity,mean_wait,mean_lift))

import numpy as np
import matplotlib.pyplot as plt


data = means_by_phase
# Nombre de trains et de phases
num_trains = len(data)
num_phases = len(data[0])

# Configuration des paramètres pour le tracé
width = 0.2  # Largeur des barres pour chaque train
x = np.arange(num_phases)  # Position des barres pour chaque phase

# Création de la figure et des sous-graphiques
fig, ax = plt.subplots()

# Parcours de chaque train
for i in range(num_trains):
    # Récupération des valeurs de chaque phase pour le train courant
    values = [data[i][j] for j in range(num_phases)]
    
    # Tracé des barres pour chaque train
    ax.bar(x + i * width, values, width, label=f"Train {i+1}")

# Configuration des axes, des légendes et du titre
ax.set_xticks(x + width * (num_trains - 1) / 2)
ax.set_xticklabels(["Phase 1", "Phase 2", "Phase 3"])
ax.legend()
ax.set_xlabel("Phases")
ax.set_ylabel("Valeur")
ax.set_title("Histogramme des valeurs de chaque train par phase")

# Affichage de la figure
plt.show()


data = means_by_phase
# Nombre de trains et de phases
num_trains = len(data)
num_phases = len(data[0])

# Configuration des paramètres pour le tracé
width = 0.2  # Largeur des barres pour chaque train
x = np.arange(num_trains)  # Position des barres pour chaque phase

# Création de la figure et des sous-graphiques
fig, ax = plt.subplots()

# Parcours de chaque train
for i in range(num_phases):
    # Récupération des valeurs de chaque phase pour le train courant
    values = [data[j][i] for j in range(num_trains)]
    
    # Tracé des barres pour chaque train
    ax.bar(x + i * width, values, width, label=f"phase {i+1}")

# Configuration des axes, des légendes et du titre
ax.set_xticks(x + width * (num_phases - 1) / 2)
ax.set_xticklabels(["train 1", "train 2", "train 3","train 4","train 5"])
ax.legend()
ax.set_ylabel("Mean rate")
ax.set_title("Histogramme des valeurs de chaque train par phase")

# Affichage de la figure
plt.show()




#%% Coefficient of variation
from elephant.statistics import isi, cv
cv_list = [cv(isi(spiketrain)) for spiketrain in spiketrains_list]


plt.figure(dpi=150)
plt.eventplot([st.magnitude for st in spiketrains_list], linelengths=0.75, linewidths=0.75, color='black')
plt.xlabel("Time, s")
plt.ylabel("Neuron id")
plt.xlim([0, 10]);

# let's plot the histogram of CVs
plt.figure(dpi=100)
plt.hist(cv_list)
plt.xlabel('CV')
plt.ylabel('count')
plt.title("Coefficient of Variation of homogeneous Poisson process")
plt.savefig(rf"{savefig_folder}/CV.svg")


#%%Optotag

concatenated_stim_idx_on=(np.where(concatenated_stim_idx == True))[0] #Where the stim is on
stim_starts=np.where(np.diff(concatenated_stim_idx_on)!=1)[0]+1
stim_starts=np.insert(stim_starts,0,0)


real_stim_idx=concatenated_stim_idx_on[stim_starts]
real_stim_times=real_stim_idx/frequency

stimulation_times=real_stim_idx/frequency          
    


for spiketrain_index,action_potential_times in enumerate(selected_spike_times):
    
    # Durée avant et après la stimulation pour la fenêtre de sélection (en secondes)
    pre_stim_duration = 0.05
    post_stim_duration = 0.25
    
    # Calcul de la durée totale
    total_duration = pre_stim_duration + post_stim_duration
    
    # Nombre de bins pour l'histogramme
    num_bins = int(total_duration * 600)  # par exemple, 10 bins par seconde
    
    # Création des limites des bins pour l'histogramme
    bin_edges = np.linspace(-pre_stim_duration, post_stim_duration, num_bins + 1)
    
    # Création du tableau 2D pour stocker les potentiels d'actions sélectionnés
    selected_potentials,selected_potentials_normalized = [],[]
    
    # Parcourir chaque temps de stimulation
    for stim_time in stimulation_times:
        # Déterminer les bornes de la fenêtre de sélection
        window_start = stim_time - pre_stim_duration
        window_end = stim_time + post_stim_duration
        
        # Sélectionner les potentiels d'actions dans la fenêtre de sélection
        potentials_in_window = action_potential_times[
            (action_potential_times >= window_start) & (action_potential_times <= window_end)
        ]
        
        
        # Normaliser les temps des potentiels par rapport à la stimulation
        normalized_potentials = potentials_in_window - stim_time
        
        # Ajouter les potentiels d'actions normalisés à selected_potentials
        selected_potentials_normalized.append(normalized_potentials)
    
        
        
        # Ajouter les potentiels d'actions à selected_potentials
        selected_potentials.append(potentials_in_window)
    
    # Convertir selected_potentials en un tableau 2D
    selected_potentials = np.array(selected_potentials)
    
    # Créer la figure et les sous-graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    plt.title(rf"Spiketrain {spiketrain_index+1}")
    
    # Tracer l'histogramme péri-stimulus sur le subplot du bas
    histograms, _ = np.histogram(action_potential_times - real_stim_times[:, None], bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax2.bar(bin_centers, histograms, width=total_duration/num_bins, align='center')
    ax2.set_ylabel('Nombre de potentiels d\'actions')
    ax2.set_title('Péri-Stimulus Histogram')
    
    
    
    # Tracer le raster plot sur le subplot du haut
    ax1.eventplot(selected_potentials_normalized, lineoffsets=0.5, linelengths=0.5, color='k')
    ax1.set_ylabel('Stimulation')
    ax1.axvspan(0,0.1,color='blue',alpha=0.3)
    
    # Normaliser l'axe des x pour le PSTH
    ax2.set_xlim(-pre_stim_duration, post_stim_duration)
    
    # Ajouter des labels pour l'axe des x
    ax2.set_xlabel('Temps (s) depuis la stimulation')
    ax2.axvspan(0,0.1,color='blue',alpha=0.3)
    
    # Ajuster les espaces entre les sous-graphiques
    plt.subplots_adjust(hspace=0.3)
    
    # Afficher la figure
    plt.show()
    
    plt.savefig(rf"{savefig_folder}/optotag_{spiketrain_name+1}.svg")
    
    
    
    
    
    
    
    
    
