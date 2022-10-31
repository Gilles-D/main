# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 14:34:52 2022

@author: Gil
"""

import numpy as np
import matplotlib.pyplot as plt
import os, re
from scipy import stats
import scipy.signal as sp

"""
PARAMETERS
"""
sampling_rate = 20000
# selected_chan=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
selected_chan=[8,10,11,13,14,15]
# selected_chan=[11,13,15]

#Filtering parameters
freq_low = 300
freq_high = 3000
order = 2

# Noise parameters
std_threshold = 5 #Times the std
noise_window = 5 #window for the noise calculation in sec
distance = 50 # distance between 2 spikes

#waveform window
waveform_window=5 #ms

filepath = r'//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/10-24/raw/0004_07_0011_20000Hz.rbf'


Plot = True
Plot_raw = True
Waveforms = True


def filter_signal(signal, order=order, sample_rate=sampling_rate, freq_low=freq_low, freq_high=freq_high, axis=0):
    import scipy.signal
    Wn = [freq_low / (sample_rate / 2), freq_high / (sample_rate / 2)]
    sos_coeff = scipy.signal.iirfilter(order, Wn, btype="band", ftype="butter", output="sos")
    filtered_signal = scipy.signal.sosfiltfilt(sos_coeff, signal, axis=axis)
    return filtered_signal


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


"""
File loading and setup
"""

raw_file = np.fromfile(filepath)
data = raw_file.reshape(int(len(raw_file)/16),-1).transpose()
time_vector = np.arange(0,len(data[0])/sampling_rate,1/sampling_rate)


filtered_signals,spikes_list,spikes_list_y,thresholds,waveforms =[],[],[],[],[]

save_path=os.path.dirname(os.path.realpath(filepath))

    
"""
Filtering preprocessing
"""

#Filter signal for each channel
for i in range(len(data)):
    if i in selected_chan:
        signal_filtered = filter_signal(data[i]) #Filter the signal
        filtered_signals.append(signal_filtered) #Append it in list
        
filtered_signals = np.array(filtered_signals) #Transform it in array

# Calculate the median signal from all filtered signals
median = np.median(filtered_signals, axis=0)

# Calculate the cmr signals for each channels
cmr_signals = filtered_signals-median     


"""
Spike detection
"""

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
    spikes_list.append(spike_times)
    spikes_list_y.append(spike_y)
    
    if Waveforms == True :
        wfs = extract_spike_waveform(signal,spike_idx)
        waveforms.append(wfs)



"""

""
Spike detection absolute (Neg + Pos)
""


waveforms_neg=[]
for signal in cmr_signals:
    abs_signal=np.absolute(signal)
    # Threshold calculation
    noise = signal[0:int(noise_window*sampling_rate)] #noise window taken from individual channel signal
    threshold = np.median(noise)+std_threshold*np.std(noise) #threshold calculation for the channel
    thresholds.append(threshold) #append it to the list regrouping threshold for each channel
    
    #Detect the spike indexes
    spike_idx, _ = sp.find_peaks(abs_signal,height=threshold,distance=distance)
    #Convert to spike times
    spike_times = spike_idx*1./sampling_rate
    #Get spikes peak 
    spike_y = signal[spike_idx]
    
    #Append spikes times to the list of all channels spikes
    spikes_list.append(spike_times)
    spikes_list_y.append(spike_y)
    
    if Waveforms == True :
        for index,i in np.ndenumerate(spike_idx):
            if spike_y[index] < threshold:
                print('neg')
                wfs_neg = extract_spike_waveform(signal,spike_idx)
            # else:
            #     print('pos')
            #     wfs = extract_spike_waveform(signal,spike_idx)
        
        waveforms_neg.append(wfs_neg)    
        # waveforms.append(wfs)  

#Transform the list of spikes in array
spikes_list= np.array(spikes_list)

#Plot waveforms for each channel
# for index,i in np.ndenumerate(waveforms):
#     plt.figure()
#     plt.title(rf'waveform_chan_{selected_chan[index[0]]}')
#     for j in i:
#         plt.plot(((np.array((range(len(j))))-25)/sampling_rate)*1000,j*1000)
#     plt.savefig(rf'{save_path}\waveform_chan_{selected_chan[index[0]]}.svg')

for i in range(len(waveforms_neg)):
    plt.figure()
    plt.title(rf'waveform_chan_neg_{selected_chan[i]}')
    for j in waveforms_neg[i]:
        plt.plot(((np.array((range(len(j))))-25)/sampling_rate)*1000,j*1000)
    plt.savefig(rf'{save_path}\waveform_neg_chan_{selected_chan[i]}.svg')
"""

"""
Plot all channel on 1 plot
"""
#Raw signal
if Plot_raw == True:
    fig, axs = plt.subplots(len(selected_chan),sharex=True,sharey=True)
    fig.suptitle('signal of all raw channels')
    # plt.setp(axs, xlim=[353.925,354.100],ylim=[-0.00353,0.00292])
    # plt.setp(axs, xlim=[25,29])
    
    for i in range(len(selected_chan)): 
        axs[i].plot(time_vector,data[i,:])
        axs[i].get_yaxis().set_visible(False)

    fig.savefig(rf'{save_path}\signal_raw.svg')

#Filtered
if Plot == True: 
    fig, axs = plt.subplots(len(selected_chan),sharex=True,sharey=True)
    fig.suptitle('signal of all channels')
    # plt.setp(axs, xlim=[353.925,354.100],ylim=[-0.00158,0.00165])
    # plt.setp(axs, xlim=[25,29],ylim=[-0.00158,0.00165])
    # plt.setp(axs,ylim=[-0.00158,0.00165])
    for i in range(len(selected_chan)):
        std = np.std(cmr_signals[i], axis=0)
        
        axs[i].axhline(-thresholds[i],color='red')
        
        axs[i].plot(time_vector,cmr_signals[i,:])
        axs[i].get_yaxis().set_visible(False)
        
        
        # spikes = spikes_list[i]
        # spikes_y = spikes_list_y[i]
        
        spikes = (spikes_list[i],spikes_list_y[i])
        for spike in spikes:
            # axs[i].axvline(spike,color='red')
            a=0
            # axs[i].scatter(spikes[0],spikes[1],label='spike detected',color='orange')
    fig.savefig(rf'{save_path}\signal.svg')

#Plot waveforms for each channel
    for index,i in np.ndenumerate(waveforms):
        plt.figure()
        plt.title(rf'waveform_chan_{selected_chan[index[0]]}')
        time_axis=np.array(range(int(-(waveform_window/1000)*20000/2),int(waveform_window/1000*20000/2)))/20000*1000
        for j in i:
            plt.plot(j*1000)
        plt.savefig(rf'{save_path}\waveform_chan_{selected_chan[index[0]]}.svg')


    
"""
Raster plot
"""
plt.figure()
plt.eventplot(spikes_list)
# plt.xlim([25,29])
plt.gca().invert_yaxis()
plt.savefig(rf'{save_path}\raster.svg')