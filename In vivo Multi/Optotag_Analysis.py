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


"""
PARAMETERS
"""
sampling_rate = 20000 #Hz
sampling_rate_stim = 10000000 #Hz
stim_window = 500 #ms



signal_file = r'//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/10-24/preprocessed/0004_07_0006_20000Hz_cmr.rbf' #use / at the end
number_of_channel=6

stim_idx_file=r'//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/stim_idx/10-24/0004_07_0006.txt'

noise_file="//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/10-24/preprocessed/0004_07_0001_20000Hz_cmr.rbf"

# # Noise parameters
# std_threshold = 5 #Times the std
# noise_window = 1 #window for the noise calculation in sec
# distance = 50 # distance between 2 spikes


Plot = True
Plot_raw = True
Waveforms = True


"""
Charger Signal
Charger Stim

Détecter spikes pour chaque channel

Pour chaque stim, déterminer si spike dans fenetre, pour chaque chan
Penser a record 30sec sans stim

raster plot pour chaque chan


"""

def load_rbf(rbf_file,sampling_rate=20000,number_of_channel=16):
    """
    
    
    """
    signal=np.fromfile(rbf_file)*1000 #in mV
    signal=signal.reshape(int(len(signal)/number_of_channel),-1).transpose()
    
    time_vector = (np.arange(0,len(signal[0])/sampling_rate,1/sampling_rate))
    
    return signal,time_vector

def load_stim_file(stim_file,sampling_rate_stim=10000000,sampling_rate=20000):
    """
    
    """
    
    stim_indexes=np.loadtxt(stim_file)
    stim_indexes_corrected = stim_indexes/(sampling_rate_stim/sampling_rate)
    
    return stim_indexes,stim_indexes_corrected

def find_peaks(signal,distance,sampling_rate,noise_window=3,std_threshold=5,thresholds=0):

    
    
    
    # Threshold calculation
    # noise = signal[0:int(noise_window*sampling_rate)] #noise window taken from individual channel signal
    # threshold = np.median(noise)+std_threshold*np.std(noise) #threshold calculation for the channel
    
    
    #Detect the spike indexes
    spike_idx, _ = sp.find_peaks(-signal,height=threshold,distance=distance) #-signal means takes negative spikes only (below threshold)
    #Convert to spike times
    spike_times = spike_idx*1./sampling_rate
    #Get spikes peak 
    spike_y = signal[spike_idx]
        
    return spike_idx,spike_times,spike_y,threshold




    
    
    # #Detect the spike indexes
    # spike_idx, _ = sp.find_peaks(-signal,height=threshold,distance=distance)
    # #Convert to spike times
    # spike_times = spike_idx*1./sampling_rate
    # #Get spikes peak 
    # spike_y = signal[spike_idx]





signal,time_vector = load_rbf(signal_file,sampling_rate,number_of_channel)
stim_idx = load_stim_file(stim_idx_file)


"""
Noise signals a disparaitre
"""

signal_noise,time_vector_noise = load_rbf(noise_file,sampling_rate,number_of_channel)
noise_chan=[]
noise_window = 10
std_threshold=5

for channel_signal in signal_noise:
    # Threshold calculation
    noise = channel_signal[0:int(noise_window*sampling_rate)] #noise window taken from individual channel signal
    threshold = np.median(noise)+std_threshold*np.std(noise) #threshold calculation for the channel
    noise_chan.append(threshold)
    
    # plt.figure()
    # plt.plot(channel_signal)
    # plt.axhline(-threshold)




"""
Spike detection
"""
spike_detected=[]

for idx, channel_signal in enumerate(signal):   
    spike_idx,spike_times,spike_y,threshold = find_peaks(channel_signal,distance=50,sampling_rate=sampling_rate,thresholds=noise_chan[idx])
    
    plt.figure()
    plt.plot(time_vector,channel_signal)
    plt.scatter(spike_times,spike_y,label='spike detected',color='orange')
    plt.axhline(-threshold)

    spike_detected.append(np.array(list(zip(spike_idx,spike_y))))


"""
Raster plot

Pour chaque channel
    Pour chaque stim
        Lister les spikes dans la fenetre
        index raster = index spike - index stim

"""


stim_idx_corrected = load_stim_file(stim_idx_file,sampling_rate_stim=sampling_rate_stim,sampling_rate=sampling_rate)[1]

spikes_rasterplot=[]

for chan, channel_signal in enumerate(signal):
    spikes_in_window = []
    
    for stim in stim_idx_corrected:
         
         for spike in spike_detected[chan]:
             
             if spike[0] > stim+100 and spike[0] <= stim+(stim_window/1000*sampling_rate):
                 spikes_in_window.append((spike[0]-stim)/sampling_rate*1000)
                 
    
    spikes_rasterplot.append(spikes_in_window)

plt.figure()
plt.eventplot(spikes_rasterplot)
           



for chan, channel_signal in enumerate(signal):
    spikes_in_window = []
    s=[]   
    for spike in spike_detected[chan]:
        s.append(spike[0])
             
    plt.figure()
    plt.plot(channel_signal)
    plt.eventplot(s)

# fig,ax=plt.subplots()
# ax.eventplot(spikes_rasterplot)
# ax.set_xticks([0,2000,4000,6000,8000,10000])
# ax.set_xticklabels([0,1,2,3,4,5])
             












"""



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
    subplot[0].axvspan(0, 1, alpha=0.1, color='blue')
    subplot[0].set_xlim(-0,25)

    width_bins = float(0.02)
    n_bins = math.ceil((max(RASTER)[0] - min(RASTER)[0])/width_bins)
    subplot[1].set_title("Peri Stimulus Histogramme {}. Bins = {}".format(manip, width_bins))
    subplot[1].set_xlabel('Time (ms)'), subplot[0].set_ylabel("Nombre d'événements")
    subplot[1].hist(np.array(RASTER), bins=n_bins, color='black')
    subplot[1].axvspan(0, 1, alpha=0.1, color='blue')
    subplot[1].set_xlim(-0,25)
    
    print(len(stim_idx))
    print(len(spike_idx))
    
    Jitter = np.mean(np.ravel(RASTER))
    Jitter_std = np.std(np.ravel(RASTER))
    Failure = 100-(len(RASTER)/len(stim_idx)*100)
    
    print (Jitter)
    print(Jitter_std)
    print(Failure)
    """