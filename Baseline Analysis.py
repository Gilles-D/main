# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:25:27 2019

@author: Gilles.DELBECQ


Baseline analysis
- Load .smr files in the subdirectories of the main folder (Baselinepath) ...Baselinepath/Date/.smr
- Detect spikes (P0)
- Detect the negative spike following each positive spike (P1)
- Get half width of each spike
- Get waveform of each spike
- Get the average of all spikes from one file

- Plot spike detection result (with half width)
- Plot Waveform of all spikes from a file
- Plot average waveform of all spikes from a file
- Plot all average waveforms fromm all the files

- Excel : Spike waveforms for each file
- Excel : Average spike waveforms of all file
- Excel : Spike times, length, half width and half width amplitude for each file

"""

from neo.io import Spike2IO as spike2
import numpy as np 
from matplotlib import pyplot as plt 
import scipy.signal as sp
from scipy import stats 
import pandas as pd
 
import os

AVERAGE_WFS = []
AVERAGE_WFS_SEM = []

#Files location
Baselinepath = 'C:/Users/Gilles.DELBECQ/Desktop/Python/smr/Baseline/Traces'
savedir = 'C:/Users/Gilles.DELBECQ/Desktop/Python/Analysis/Baseline'

#Params for spike detection
threshold = 0.4
distance = 10
negdistance = 30

#File loading list loop
List_File_paths = []
for r, d, f in os.walk(Baselinepath):
# r=root, d=directories, f = files
    for filename in f:
        if '.smr' in filename:
            List_File_paths.append(os.path.join(r, filename))

            
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

def negative_spikes(spike_idx):
    '''
    Extract the negative Spike (P1) following the positive spikes given by their index
    '''
    #Detect the neg spikes indexes
    neg_spike_times_list = []
    neg_spike_y_list = []
    for i in range(len(spike_idx)): 
        neg_spike_wf = signal[spike_idx[i] : spike_idx[i]+negdistance] #Take a window from P0 to P0+negdistance
        neg_spike_idx, _ = sp.find_peaks(-neg_spike_wf, distance=negdistance) #Find the neg peak
        neg_spike_y = neg_spike_wf[neg_spike_idx] #Take the amplitude value of the neg peak
        neg_spike_times = neg_spike_idx*1./sampling_rate #convert the index of the peak in time (in s)
        neg_spike_times_list.append(neg_spike_times) #Append in the list
        neg_spike_y_list.append(neg_spike_y) #Append the amplitude value
    neg_spike_times_list = np.array(neg_spike_times_list).flatten() #extract a one-dimensional array from your list of multi-dimensional arrays
    neg_spike_y_list = np.array(neg_spike_y_list).flatten()
    return neg_spike_times_list, neg_spike_y_list

def spike_half_width(signal, spike_idx, rel_height=0.5):
    '''
    Get the half width of all the spikes given by their index
    signal (1-d array) = the ephy signal
    spike_idx (1-d array or integer list) : array containing the spike indexes (in points)
    rel_height : float, Chooses the relative height at which the peak width is measured as a percentage of its prominence. 1.0 calculates the width of the peak at its lowest contour line while 0.5 evaluates at half the prominence height. Must be at least 0. See notes for further explanation.
    '''
    spike_halfwidth = sp.peak_widths(signal, spike_idx, rel_height=0.5)
    #Convert width, xmin & xmax in ms
    half_width = spike_halfwidth[0]*1./sampling_rate
    half_amplitude = spike_halfwidth[1]
    half_xmin = spike_halfwidth[2]*1./sampling_rate
    half_xmax = spike_halfwidth[3]*1./sampling_rate
    return half_amplitude, half_xmin, half_xmax, half_width
    
for file_path in List_File_paths:
    '''--------Initialization--------'''
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


    '''--------Spike detection--------'''          
    #Positive Spike P0 and the following Negative spike P1
    spike_idx, _ = sp.find_peaks(signal,height=threshold,distance=distance) #Detect the spike indexes
    spike_times = spike_idx*1./sampling_rate #Convert to spike times
    spike_y = signal[spike_idx] #Get spikes peak 
    neg_spikes = negative_spikes(spike_idx) #Detect the negative spike following each positive spikes

    '''--------Spike half width--------'''  
    spike_half_width_list = spike_half_width(signal, spike_idx, rel_height=0.5) #Get the half width of all the spikes

    '''--------Waveforms extraction of all spikes--------'''    
    spike_wfs = extract_spike_waveform(signal, spike_idx,left_width=10,right_width=30)
    spike_wfs_indexes = np.arange(-10,30,1)
    spike_wfs_times = spike_wfs_indexes*1./sampling_rate*1000 #in ms

    '''--------Waveform average of the spikes of a file--------'''   
    spike_wfs_average = np.average(spike_wfs, axis=0)
    spike_wfs_average_sem = stats.sem(spike_wfs, ddof=len(spike_wfs)-2)
    AVERAGE_WFS.append(spike_wfs_average)
    AVERAGE_WFS_SEM.append(spike_wfs_average_sem)

    '''------- Plots -------- '''
    '''Plot Spike detection result'''
    # Do a figure to show spike detection result
    plt.figure()
    plt.title('Spike detection Manip {} {} (thres={},dist={})'.format(date, exp, threshold, distance))
    plt.plot(time_vector, signal, label='Data')
    plt.hlines(spike_half_width_list[0], spike_half_width_list[1], spike_half_width_list[2], color="C2") #Draw half width of each spikes
    plt.scatter(spike_times,spike_y,label='spike detected',color='orange')
    plt.xlabel('Time (s)') ; plt.ylabel('Signal ({})'.format(units[-2:]))   
    
    '''Plot all the waveforms and the average waveform of a file'''
    fig, subplot = plt.subplots(2,1,sharex=True) #subplots (lignes,colonnes,sharex=true pour partager le même axe x entre les plots)
    subplot[0].set_title('All the waveforms Manip {} {}'.format(date, exp))
    subplot[0].set_xlabel('Time (ms)'), subplot[0].set_ylabel('Signal ({})'.format(units[-2:]))
    subplot[0].scatter(neg_spikes[0]*1000,neg_spikes[1],label='negspike detected',color='red')
    for wf in spike_wfs:
        subplot[0].plot(spike_wfs_times,wf,alpha=0.8)
    
    subplot[1].set_title('Waveform average Manip {} {}'.format(date, exp))
    subplot[1].set_xlabel('Time (ms)'), subplot[1].set_ylabel('Signal ({})'.format(units[-2:]))
    subplot[1].plot(spike_wfs_times,spike_wfs_average,alpha=0.8)
    subplot[1].errorbar(spike_wfs_times,spike_wfs_average,yerr=spike_wfs_average_sem)
    
    
    
    ''' -----------Save as excel----------'''
    '''Save the spikes waveforms of the neuron'''
    df = pd.DataFrame(np.asarray(spike_wfs))
    df.to_excel('{}/{}_{}_waveforms.xlsx'.format(savedir, date, exp)) 

    '''Save the spike waveform average of the neuron'''
    dfx = pd.DataFrame(np.asarray(spike_wfs_average))
    dfx.to_excel('{}/{}_{}_waveformsaverage.xlsx'.format(savedir, date, exp))   

    '''Save Spike times, relative neg spike time (spike length), half width and half width amplitude'''
    data_critere_spike = { 'Spike_times' : spike_times, 'Spike_length' : neg_spikes[0], 'Spike_half_width' : spike_half_width_list[3], 'Spike_half_width_amplitude' : spike_half_width_list[0]}
    df5 = pd.DataFrame(data_critere_spike)
    df5.to_excel('{}/{}_{}_Time_criteria.xlsx'.format(savedir, date, exp))
    
'''Plot all the average waveform from all files'''
np.asarray(AVERAGE_WFS)
AVERAGE_WFS_transposed = np.transpose(AVERAGE_WFS)
np.asarray(AVERAGE_WFS_SEM)
plt.figure()
plt.title('Waveforms averages')
plt.xlabel('Time (ms)'), plt.ylabel('Signal ({})'.format(units[-2:]))
plt.plot(AVERAGE_WFS_transposed,alpha=0.8)
plt.errorbar(spike_wfs_times,AVERAGE_WFS,yerr=AVERAGE_WFS_SEM)