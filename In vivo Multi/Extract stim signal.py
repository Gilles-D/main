# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 15:28:47 2022

@author: Gilles.DELBECQ
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal as sp



'''
Parameters
'''
sampling_rate = 20000

sampling_rate_stim = 1000000

signal_file = r'//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/10-24/preprocessed/0004_07_0005_20000Hz_cmr.rbf' #use / at the end
number_of_channel=6

stim_idx_file=r'//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/stim_idx/10-24/0004_07_0005.txt'

Save=True



#waveform window
waveform_window=300 #ms


noise_window=50
std_threshold = 5 #Times the std
distance = 50 # distance between 2 spikes




"""
Functions
"""





def extract_waveform(signal_file, stim_idx_file, waveform_window):
    
    '''
    Function to extract spikes waveforms in spike2 recordings
    
    INPUTS :

    
    OUTPUTS : 

    
    '''
    
    waveforms=[]
    
    signal=np.fromfile(signal_file)*1000 #in mV
    signal=signal.reshape(int(len(signal)/number_of_channel),-1).transpose()
    
    stim_indexes=np.loadtxt(stim_idx_file)
    stim_indexes_corrected = stim_indexes/(sampling_rate_stim/sampling_rate)

    
    """
    Drop le premier index ?
    
    stim_indexes[5]-stim_indexes[4] = 1049300 ou 1049350 = 1 sec
    
    """
    
       
    window_idx=int((waveform_window/1000)*20000) 
    
    for chan in range(number_of_channel):
        chan_signal=signal[chan]
        chan_wf=[]
        
        for i in range(len(stim_indexes_corrected)): 
            index = stim_indexes_corrected[i]

            stim_wf = chan_signal[int(index) : int(index+window_idx)]
    
            chan_wf.append(stim_wf)
            
        waveforms.append(chan_wf)
    
    time_vector = (np.arange(0,window_idx/sampling_rate,1/sampling_rate))*1000
    
    return waveforms,time_vector



# plt.plot(signal[3])

# for stim in stim_indexes_corrected:
#     plt.axvline(stim, c='r')

waveforms,time_vector = extract_waveform(signal_file,stim_idx_file,waveform_window)

for chan in waveforms:
    SPIKE_Y=[]
    SPIKE_X=[]
    
    plt.figure()
    med=np.median(chan, axis=0)
    for stim in chan:
        plt.plot(time_vector,stim,alpha=0.5)
        
        # Threshold calculation
        noise = stim[-int(noise_window*sampling_rate):] #noise window taken from individual channel signal
        threshold = np.median(noise)+std_threshold*np.std(noise) #threshold calculation for the channel
        
        stim_tronc=stim
        #Detect the spike indexes
        spike_idx, _ = sp.find_peaks(-stim_tronc,height=threshold,distance=distance)
        #Convert to spike times
        spike_times = spike_idx*1./sampling_rate*1000
        #Get spikes peak 
        spike_y = stim[spike_idx]
        
        SPIKE_Y.append(spike_y)
        SPIKE_X.append(spike_times)
        

        
        
    
for chan in waveforms:
    # Threshold calculation
    noise = chan[-int(noise_window*sampling_rate):] #noise window taken from individual channel signal
    threshold = np.median(noise)+std_threshold*np.std(noise) #threshold calculation for the channel

    #Detect the spike indexes
    spike_idx, _ = sp.find_peaks(-chan,height=threshold,distance=distance)
    #Convert to spike times
    spike_times = spike_idx*1./sampling_rate
    #Get spikes peak 
    spike_y = chan[spike_idx]
    
    #Append spikes times to the list of all channels spikes
    spikes_list.append(spike_times)
    spikes_list_y.append(spike_y)
    
    if Waveforms == True :
        wfs = extract_spike_waveform(signal,spike_idx)
        waveforms.append(wfs)
