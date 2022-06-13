# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 12:22:14 2021

@author: Gilles
"""

from neo.io import Spike2IO as spike2
import numpy as np 
from matplotlib import pyplot as plt 
import scipy.signal as sp
from scipy import stats 
import pandas as pd
import os

window_before = 0.2
window_after = 0.5

def moving_average(a, n=100) : 
    '''
    performs moving average/Running average/rolling average.
    Creates a series of averages of different subsets of the full data set
    Inputs: 
        a = signal (1D array) : the raw signal
        n = Number of values to be averaged
    Output:
        Averaged signal
        
    '''
    
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


#Files location
data_dir = r'C:/Users/MOCAP/Desktop/16-03-21/Nouveau dossier'


#File loading list loop
List_File_paths = []
for r, d, f in os.walk(data_dir):
# r=root, d=directories, f = files
    for filename in f:
        if '.smr' in filename:
            List_File_paths.append(os.path.join(r, filename))
            
for file_path in List_File_paths:
    '''--------Initialization--------'''

    #File identification
    exp = os.path.splitext(os.path.basename(file_path))[0]
    print(exp)
    
    #Load file and block
    reader = spike2(file_path)
    block = reader.read_block()
    
    #Get signal, time_vector and sampling rate
    signal = np.ravel(block.segments[0].analogsignals[1].magnitude)
    time_vector = np.ravel(block.segments[0].analogsignals[1].times)
    sampling_rate = float(block.segments[0].analogsignals[1].sampling_rate)
    units = str(block.segments[0].analogsignals[1].units) 

    #Load stim times
    stim_idx = reader.get_event_timestamps(event_channel_index=1)
    stim_idx = np.array(stim_idx[0])
    stim_times = stim_idx / 500000.00000000006
    stim_idx_corrected=stim_times*sampling_rate
    
 
    #     #Plot unfiltered
    # plt.figure()
    # plt.plot(time_vector,signal)
    # for stim_time in stim_times:
    #     plt.axvline(x=stim_time )
    # plt.title(exp)
    
    # #Plot filtered
    # plt.plot(time_vector[99:],moving_average(signal))
    

    signal_chunks = list()

    for i in range(0, len(stim_times)):
        signal_chunks.append(signal[int(stim_idx_corrected[i])- int(window_before*sampling_rate) : int(stim_idx_corrected[i]) + int(window_after*sampling_rate)])
  
    signal_chunks_filtered = list()

    for i in range(0, len(stim_times)):
        signal_chunks_filtered.append(moving_average(signal[int(stim_idx_corrected[i]-99)- int(window_before*sampling_rate) : int(stim_idx_corrected[i]-99) + int(window_after*sampling_rate)]))


    event_triggered_average = np.mean(signal_chunks, axis = 0)
    x = list(range(len(event_triggered_average)))
    x = [i/sampling_rate for i in x]
    
    plt.figure()
    plt.plot(x,event_triggered_average)
    plt.title("Average {}".format(exp))
    plt.axvline(x=0.05+window_before, color='green')
    plt.axvline(x=0.095+window_before, color='green')
    
    event_triggered_average_filtered = np.mean(signal_chunks_filtered, axis = 0)
    
    x = list(range(len(event_triggered_average_filtered)))
    x = [i/sampling_rate for i in x]
    plt.plot(x,event_triggered_average_filtered)
    
    plt.savefig("Average {}".format(exp))

    
    # windows=[]
    # signal_windowed=[]
    # # plt.figure()
    # for stim_time in stim_times:
    #     windows.append([stim_time,stim_time+window_length])
    
        # window_timing=[x for x in time_vector if x >= stim_time and x <= stim_time+window_length]
    # plt.plot(time_vector,signal)
    # plt.title("Average {}".format(exp))