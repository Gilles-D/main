# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:31:37 2022

@author: Gilles.DELBECQ
"""


import numpy as np
import matplotlib.pyplot as plt
import os, re
from scipy import stats
import scipy.signal as sp
import pandas as pd

"""
PARAMETERS
"""

sampling_rate = 20000

selected_chan=[8,10,11,13,14,15]
Animal="0004"

freq_low = 300
freq_high = 3000
order = 2

folderpath = r'//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/Analysis/PCA test/files/preprocessed/' #use / everywhere

# Noise parameters
std_threshold = 5 #Times the std
noise_window = 5 #window for the noise calculation in sec
distance = 50 # distance between 2 spikes

#waveform window
waveform_window=2 #ms

Plot = True
Plot_raw = True
Waveforms = True





whole_cmr_signal3,whole_cmr_signal2=np.array([]),np.array([])

def extract_spike_waveform(signal, spike_idx, left_width=(waveform_window/1000)*20000/2, right_width=(waveform_window/1000)*20000/2):
    
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

def threshold(signal,noise_window=5,sampling_rate=20000):
    noise = signal[0:int(noise_window*sampling_rate)] #noise window taken from individual channel signal
    threshold = np.median(noise)+std_threshold*np.std(noise) #threshold calculation for the channel
    
    return threshold

list_files=[]
for path, subdirs, files in os.walk(folderpath):
    for name in files:
        list_files.append(os.path.join(path, name))

for file in list_files:
    #load signals
    data_cmr=np.fromfile(file)*1000 #in mV
    data_cmr=data_cmr.reshape(int(len(data_cmr)/6),-1).transpose()
    
    #select channel 3
    data_cmr_chan3=data_cmr[3]
    #select channel 0
    data_cmr_chan2=data_cmr[2]
    
    #append channel 3 signal
    whole_cmr_signal3 = np.concatenate((whole_cmr_signal3,data_cmr_chan3))
    #append channel 2 signal
    whole_cmr_signal2 = np.concatenate((whole_cmr_signal2,data_cmr_chan2))

#Time vector on the whole appended signals
time_vector = np.arange(0,len(whole_cmr_signal3)/sampling_rate,1/sampling_rate)#in seconds



#Detect the spike indexes
spike_idx, _ = sp.find_peaks(-whole_cmr_signal3,height=threshold(whole_cmr_signal3),distance=distance)
#Convert to spike times
spike_times = spike_idx*1./sampling_rate#in seconds
#Get spikes peak 
spike_y = whole_cmr_signal3[spike_idx]

#get waveforms
wfs = extract_spike_waveform(whole_cmr_signal3,spike_idx)

wfs_2 = extract_spike_waveform(whole_cmr_signal2,spike_idx)

#get width and half-width
spike_width=sp.peak_widths(-whole_cmr_signal3, spike_idx, rel_height=1)
spike_halfwidth = sp.peak_widths(-whole_cmr_signal3, spike_idx, rel_height=0.5)
#setup the parameters in an array
parameters = np.column_stack((spike_width[0],spike_width[1],spike_halfwidth[0],spike_halfwidth[1]))


#Save arrays as excel files
waveforms_df = pd.DataFrame(wfs)
parameters_df = pd.DataFrame(parameters)
spike_times_df= pd.DataFrame(spike_times)

with pd.ExcelWriter(r"\\equipe2-nas1\Gilles.DELBECQ\Data\ePhy\Cohorte 1\Analysis\PCA test\waveforms_chan3.xlsx") as writer:
    waveforms_df.to_excel(writer)
    
with pd.ExcelWriter(r"\\equipe2-nas1\Gilles.DELBECQ\Data\ePhy\Cohorte 1\Analysis\PCA test\parameters_chan3.xlsx") as writer:
    parameters_df.to_excel(writer)    

with pd.ExcelWriter(r"\\equipe2-nas1\Gilles.DELBECQ\Data\ePhy\Cohorte 1\Analysis\PCA test\spike_times.xlsx") as writer:
    spike_times_df.to_excel(writer)   




waveforms2_df = pd.DataFrame(wfs_2)


with pd.ExcelWriter(r"\\equipe2-nas1\Gilles.DELBECQ\Data\ePhy\Cohorte 1\Analysis\PCA test\waveforms_chan2.xlsx") as writer:
    waveforms_df.to_excel(writer)



wfs_all = np.column_stack((wfs,wfs_2))
wfs_all_df = pd.DataFrame(wfs_all)
with pd.ExcelWriter(r"\\equipe2-nas1\Gilles.DELBECQ\Data\ePhy\Cohorte 1\Analysis\PCA test\waveforms_all.xlsx") as writer:
    wfs_all_df.to_excel(writer)


"""    

fig, axs = plt.subplots(1)

# std = np.std(cmr_signals[i], axis=0)
plt.axhline(-threshold(whole_cmr_signal3),color='red')
plt.plot(time_vector,whole_cmr_signal3)
        # spikes = spikes_list[i]
        # spikes_y = spikes_list_y[i]
plt.scatter(spike_times,spike_y,label='spike detected',color='orange')

plt.figure()
for i in wfs:
    plt.plot(i)
    
plt.figure()    
plt.eventplot(spike_times)

cumsum = stats.cumfreq(spike_times)
histo = np.histogram(spike_times)
cumulative_histo_counts = histo[0].cumsum()
bin_size = histo[1][1]-histo[1][0]
plt.bar(histo[1][:-1], cumulative_histo_counts, width=bin_size)
"""