# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:37:54 2019

@author: ludovic.spaeth
"""

from neo.io import Spike2IO as spike2
import numpy as np 
from matplotlib import pyplot as plt 
import scipy.signal as sp 
import pandas as pd 

#File location
file = 'D:/Analyses/Fig E.1/Figure/35.smr'
savedir = 'D:/Analyses/Fig E.1/Figure/'
manip = '35'

#Params for spike detection
threshold = 0.31
distance = 10

#Load file and block
reader = spike2(file)
block = reader.read_block()

#Get signal, time_vector and sampling rate
signal = np.ravel(block.segments[0].analogsignals[0].magnitude)
time_vector = np.ravel(block.segments[0].analogsignals[0].times)
sampling_rate = float(block.segments[0].analogsignals[0].sampling_rate)

units = str(block.segments[0].analogsignals[0].units)


#Detect the spike indexes
spike_idx, _ = sp.find_peaks(signal,height=threshold,distance=distance)
#Convert to spike times
spike_times = spike_idx*1./sampling_rate
#Get spikes peak 
spike_y = signal[spike_idx]


#Do a figure to show spike detection result
plt.figure()
plt.title('Spike detection (thres={},dist={})'.format(threshold, distance))
plt.plot(time_vector, signal, label='Data')
plt.scatter(spike_times,spike_y,label='spike detected',color='orange')
plt.xlabel('Time (s)') ; plt.ylabel('Signal ({})'.format(units[-2:]))


def extract_spike_waveform(signal, spike_idx, left_width, right_width):
    
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


spike_wfs = extract_spike_waveform(signal, spike_idx,left_width=10,right_width=30)
spike_wfs_indexes = np.arange(-10,30,1)
spike_wfs_times = spike_wfs_indexes*1./sampling_rate*1000 #in ms

#Plot the waveforms 
plt.figure()
plt.title('All the waveforms')
plt.xlabel('Time (ms)'), plt.ylabel('Signal ({})'.format(units[-2:]))
for wf in spike_wfs:
    plt.plot(spike_wfs_times,wf,alpha=0.8)
    
    
##Save the spikes waveforms to excel
#df = pd.DataFrame(np.asarray(spike_wfs))
#df.to_excel('{}/{}_waveforms.xlsx'.format(savedir,manip))    

#Save the spike times to excel
df2 = pd.DataFrame(spike_times)
df2.to_excel('{}/{}_spike_times.xlsx'.format(savedir,manip))    

