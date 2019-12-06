# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 10:35:21 2019

@author: Gilles.DELBECQ

Stimulation analysis
- Load .smr files in the subdirectories of the main folder (Simulationpath) ...Simulationpath/Date/.smr
- Detect spikes and get stimulation times
- 

"""


from neo.io import Spike2IO as spike2
import numpy as np 
from matplotlib import pyplot as plt 
import scipy.signal as sp
from scipy import stats 
import pandas as pd
import os

#Files location
Stimulationpath = 'C:/Users/Gilles.DELBECQ/Desktop/Python/smr/Stim/'
savedir = 'C:/Users/Gilles.DELBECQ/Desktop/Python/Analysis/Stim'

#Params for spike detection
threshold = 0.4
distance = 10
fenetre = 20/1000


#File loading list loop
List_File_paths = []
for r, d, f in os.walk(Stimulationpath):
# r=root, d=directories, f = files
    for filename in f:
        if '.smr' in filename:
            List_File_paths.append(os.path.join(r, filename))
            
for file_path in List_File_paths:
    '''--------Initialization--------'''
    RASTER = []
    N = []
    n = 0
    #File identification
    exp = os.path.splitext(os.path.basename(file_path))[0]
    date = os.path.basename(os.path.dirname(file_path))
    print('Experiment : {} {}' .format(date, exp))
    
    #Load file and block
    reader = spike2(file_path)
    block = reader.read_block()
    
    #Get signal, time_vector and sampling rate
    signal = np.ravel(block.segments[0].analogsignals[0].magnitude)
    time_vector = np.ravel(block.segments[0].analogsignals[0].times)
    sampling_rate = float(block.segments[0].analogsignals[0].sampling_rate)
    units = str(block.segments[0].analogsignals[0].units) 

    #Load stim times
    stim_idx = reader.get_event_timestamps(event_channel_index=1)
    stim_idx = np.array(stim_idx[0])
    stim_times = stim_idx / 500000.00000000006
    
    '''--------Spike detection--------'''          
    #Positive Spike P0 and the following Negative spike P1
    spike_idx, _ = sp.find_peaks(signal,height=threshold,distance=distance) #Detect the spike indexes
    spike_times = spike_idx*1./sampling_rate #Convert to spike times
    spike_y = signal[spike_idx] #Get spikes peak 
    
    '''--------Raster Plot-----'''
    for stim_time in stim_times:
        n = n + 1
        for spike_time in spike_times:
            if spike_time > stim_time and spike_time <= stim_time+fenetre or spike_time < stim_time and spike_time >= stim_time-fenetre:
                spike_raster = spike_time - stim_time
                N.append([n])
                RASTER.append([spike_raster*1000])
    
    '''------- Plots -------- '''
    # print(RASTER)
    plt.figure()
    plt.eventplot(RASTER, lineoffsets = 0.425, linelengths = 1, colors = 'black')
    plt.title('RasterPlot')
    plt.xlabel('Temps en ms')
    plt.ylabel("Nombre d'événements")
    plt.axvspan(0, 5, alpha=0.1, color='blue')
    plt.xlim(-19,22)
    plt.ylim(-9,115)
    # plt.savefig("D:/Analyses/Fig E.1/Figure/RASTER32.svg", transparent=True)
    
    
    bins = 200
    plt.figure()
    plt.hist(np.array(RASTER), bins=bins, color='black')
    plt.title("Peri Stimulus Histogramme. Bins = {}".format(bins))
    plt.axvspan(0, 5, alpha=0.1, color='blue')
    plt.xlim(-19,22)
    plt.ylim(0,105)
    plt.show()
    # plt.savefig("D:/Analyses/Fig E.1/Figure/PSTH32.svg", transparent=True)
    
    '''Plot Spike detection result with stimulations'''
    # Do a figure to show spike detection result
    plt.figure()
    plt.title('Spike detection Manip {} {} (thres={},dist={})'.format(date, exp, threshold, distance))
    plt.plot(time_vector, signal, label='Data')
    plt.scatter(spike_times,spike_y,label='spike detected',color='orange')
    plt.xlabel('Time (s)') ; plt.ylabel('Signal ({})'.format(units[-2:]))
    for stim in stim_times:
        plt.axvline(stim, c= 'red') 
    
    fail = 0
    success = 0
    for x in range(len(stim_times)):
        k=0
        stim_n = stim_times[x]
        for i in range(len(spike_times)):
            spike_i = spike_times[i]
            if spike_i > stim_n and spike_i <= stim_n+20 :
                k = k+1
        if k == 0:
            fail+=1
            print(stim_n)
        else:
            success+=1
    mean = fail/(fail+success)*100
    print('Failure rate = {}'.format(mean))   