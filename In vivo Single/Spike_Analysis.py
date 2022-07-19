# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:42:34 2022

@author: gilles.DELBECQ
"""

from neo.io import Spike2IO as spike2
import numpy as np 
from matplotlib import pyplot as plt 
import scipy.signal as sp
from scipy import stats 
import pandas as pd 
import os
import math


#Files location
files_path = r'D:\Working_Dir\Ephy\Anesth\Optotag cx\Thy1\19-07-22'
savedir = r'D:\Working_Dir\Ephy\Anesth\Optotag cx\Thy1\19-07-22'

"""
PARAMETERS
"""

# Noise parameters
std_threshold = 7 #Times the std
noise_window = 1 #window for the noise calculation in sec
distance = 100 # distance between 2 spikes


Plot = True
Plot_raw = True
Waveforms = True


List_File_paths = []
for r, d, f in os.walk(files_path):
# r=root, d=directories, f = files
    for filename in f:
        if '.smr' in filename and '.smrx' not in filename:
            List_File_paths.append(os.path.join(r, filename))



def extract_spike_waveform(signal, spike_idx, left_width=25, right_width=25):
    
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



def get_signal(file_path):
    #Load file and block
    reader = spike2(file_path)
    block = reader.read_block()
    
    #Get signal, time_vector and sampling rate
    signal = np.ravel(block.segments[0].analogsignals[0].magnitude)
    time_vector = np.ravel(block.segments[0].analogsignals[0].times)
    sampling_rate = float(block.segments[0].analogsignals[0].sampling_rate)
    
    units = str(block.segments[0].analogsignals[0].units)
    


    return signal,time_vector,sampling_rate,units



def get_stim(file_path):
    #Load file and block
    reader = spike2(file_path)
    block = reader.read_block()
    
    
    #Load stim times
    stim_idx = reader.get_event_timestamps(event_channel_index=1)
    stim_idx = np.array(stim_idx[0])
    stim_times = stim_idx / 500000.00000000006
    stim_idx_corrected=stim_times*sampling_rate
    
    return stim_idx,stim_times,stim_idx_corrected


for file_path in List_File_paths:
    manip = os.path.splitext(os.path.basename(file_path))[0]
    date = os.path.basename(os.path.dirname(file_path))
    print('Manip {} {}' .format(date, manip))
    
    save_path=os.path.dirname(os.path.realpath(file_path))

    
    #Get signal, time_vector and sampling rate
    signal,time_vector,sampling_rate,units = get_signal(file_path)
    

    
    abs_signal=np.absolute(signal)
     
    noise = abs_signal[0:int(noise_window*sampling_rate)] #noise window taken from individual channel signal
    threshold = np.median(noise)+std_threshold*np.std(noise) #threshold calculation for the channel
    
    '''Positive or Negative spikes ?'''
    #Detect the spike indexes
    spike_idx, _ = sp.find_peaks(abs_signal,height=threshold,distance=distance)  
    #Convert to spike times
    spike_times = spike_idx*1./sampling_rate
    #Get spikes peak 
    spike_y = signal[spike_idx]
    
    if np.median(spike_y) < threshold:
        #Detect the negative spike indexes
        spike_idx, _ = sp.find_peaks(-signal,height=threshold,distance=distance)  
        #Convert to spike times
        spike_times = spike_idx*1./sampling_rate
        #Get spikes peak 
        spike_y = signal[spike_idx]
    else:
        #Detect the positive spike indexes
        spike_idx, _ = sp.find_peaks(signal,height=threshold,distance=distance)  
        #Convert to spike times
        spike_times = spike_idx*1./sampling_rate
        #Get spikes peak 
        spike_y = signal[spike_idx]



    '''Extraction des waveform'''  
    spike_wfs = extract_spike_waveform(signal, spike_idx,left_width=50,right_width=80)
    spike_wfs_indexes = np.arange(-50,80,1)
    spike_wfs_times = spike_wfs_indexes*1./sampling_rate*1000 #in ms


    stim_idx,stim_times,stim_idx_corrected=get_stim(file_path)
    window = (100/1000)*sampling_rate
    RASTER=[]
    
    for idx in stim_idx_corrected:
        # print(idx)
        a=0
        for s_idx in spike_idx:
            if s_idx > idx and s_idx <= idx+window and a==0:
                spike_raster = s_idx - idx
                RASTER.append([spike_raster*1000/sampling_rate])
                a=1
    


    '''------- Plots -------- '''
    '''Plot figures spike detection'''
    # Do a figure to show spike detection result
    plt.figure()
    plt.title('Spike detection Manip {} {} (thres std={},dist={})'.format(date, manip, std_threshold, distance))
    plt.plot(time_vector, signal, label='Data')
    plt.scatter(spike_times,spike_y,label='spike detected',color='orange')
    plt.xlabel('Time (s)') ; plt.ylabel('Signal ({})'.format(units[-2:]))   
    
        
    '''Plot all the waveforms'''
    figure = plt.figure()
    plt.title('Waveforms Manip {} {} (thres std={},dist={})'.format(date, manip, std_threshold, distance))
    for wf in spike_wfs:
        plt.plot(spike_wfs_times,wf,alpha=0.8)
        
    
    
    '''Plot Raster'''
    fig, subplot = plt.subplots(2,1,sharex=True) #subplots (lignes,colonnes,sharex=true pour partager le même axe x entre les plots)
    subplot[0].set_title('RasterPlot {}'.format(manip))
    subplot[0].set_ylabel("Nombre d'événements")
    subplot[0].eventplot(RASTER, lineoffsets = 1, linelengths = 0.5, colors = 'black')
    subplot[0].axvspan(0, 2, alpha=0.1, color='blue')
    subplot[0].set_xlim(-0,25)

    width_bins = float(0.02)
    n_bins = math.ceil((max(RASTER)[0] - min(RASTER)[0])/width_bins)
    subplot[1].set_title("Peri Stimulus Histogramme {}. Bins = {}".format(manip, width_bins))
    subplot[1].set_xlabel('Time (ms)'), subplot[0].set_ylabel("Nombre d'événements")
    subplot[1].hist(np.array(RASTER), bins=n_bins, color='black')
    subplot[1].axvspan(0, 2, alpha=0.1, color='blue')
    subplot[1].set_xlim(-0,25)
    
    print(len(stim_idx))
    print(len(spike_idx))
    