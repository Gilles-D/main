# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:37:54 2019

@author: ludovic.spaeth
Modifié par Gilles

Lit l'ensemble des fichiers .smr des sous dossiers (date) du dossier data
détecte les spikes de la baseline
détecte le spike negatif suivant
mesure la durée demi pic
Output :
    spike time de chaque spike positif
    la durée entre Pic max et pic min
    durée de demi pic
    waveform de chaque spike

"""

from neo.io import Spike2IO as spike2
import numpy as np 
from matplotlib import pyplot as plt 
import scipy.signal as sp
from scipy import stats 
import pandas as pd 
import os


#Files location
Baselinepath = 'D:/Analyses\Baseline\Traces'
savedir = 'D:/Analyses\Baseline\data2'
    
List_File_paths = []
for r, d, f in os.walk(Baselinepath):
# r=root, d=directories, f = files
    for filename in f:
        if '.smr' in filename:
            List_File_paths.append(os.path.join(r, filename))

AVERAGE_WFS = []
AVERAGE_WFS_SEM = []
    
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

for file_path in List_File_paths:
    manip = os.path.splitext(os.path.basename(file_path))[0]
    date = os.path.basename(os.path.dirname(file_path))
    print('Manip{} {}' .format(date, manip))
    #Params for spike detection
    threshold = 0.4
    distance = 10
    negdistance = 30
    
    #Load file and block
    reader = spike2(file_path)
    block = reader.read_block()
    
    #Get signal, time_vector and sampling rate
    signal = np.ravel(block.segments[0].analogsignals[0].magnitude)
    time_vector = np.ravel(block.segments[0].analogsignals[0].times)
    sampling_rate = float(block.segments[0].analogsignals[0].sampling_rate)
    
    units = str(block.segments[0].analogsignals[0].units)
          
    '''Positive Spikes P0'''
    #Detect the spike indexes
    spike_idx, _ = sp.find_peaks(signal,height=threshold,distance=distance)
    #Convert to spike times
    spike_times = spike_idx*1./sampling_rate
    #Get spikes peak 
    spike_y = signal[spike_idx]
    
    '''Negative Spikes P1'''
    #Detect the neg spikes indexes
    NEGSPIKES_TIMES = []
    NEGSPIKES_TIMES_ms = []
    NEGSPIKES_Y = []
    for i in range(len(spike_idx)): 
        index = spike_idx[i]
        negspike_wf = signal[index : index+negdistance] #Take a window from P0 to P0+negdistance
        negspike_idx, _ = sp.find_peaks(-negspike_wf, distance=negdistance) #Find the neg peak
        negspike_y = negspike_wf[negspike_idx] #Take the amplitude value of the neg peak
        negspike_times = negspike_idx*1./sampling_rate #convert the index of the peak in time (in s)
        NEGSPIKES_TIMES.append(negspike_times) #Append in the list
        NEGSPIKES_TIMES_ms.append(negspike_times*1000)# Append the converted time in ms (for the waveform plot)    
        NEGSPIKES_Y.append(negspike_y) #Append the amplitude value
    NEGSPIKES_TIMES = np.array(NEGSPIKES_TIMES).flatten() #extract a one-dimensional array from your list of multi-dimensional arrays
    NEGSPIKES_TIMES_ms = np.array(NEGSPIKES_TIMES_ms).flatten()
    NEGSPIKES_Y = np.array(NEGSPIKES_Y).flatten()
    
    '''Spike half width Critère B'''
    CritB = sp.peak_widths(signal, spike_idx, rel_height=0.5)
    #Convert width, xmin & xmax in ms
    half_width = CritB[0]*1./sampling_rate
    half_amplitude = CritB[1]
    half_xmin = CritB[2]*1./sampling_rate
    half_xmax = CritB[3]*1./sampling_rate
    
    '''Extraction des waveform'''  
    spike_wfs = extract_spike_waveform(signal, spike_idx,left_width=10,right_width=30)
    spike_wfs_indexes = np.arange(-10,30,1)
    spike_wfs_times = spike_wfs_indexes*1./sampling_rate*1000 #in ms
  
    '''Waveform average of the spikes of a neuron'''
    spike_wfs_average = np.average(spike_wfs, axis=0)
    spike_wfs_average_sem = stats.sem(spike_wfs, ddof=len(spike_wfs)-2)
    AVERAGE_WFS.append(spike_wfs_average)
    AVERAGE_WFS_SEM.append(spike_wfs_average_sem)
    
    '''------- Plots -------- '''
    '''Plot figures spike detection'''
    # Do a figure to show spike detection result
    plt.figure()
    plt.title('Spike detection Manip {} {} (thres={},dist={})'.format(date, manip, threshold, distance))
    plt.plot(time_vector, signal, label='Data')
    plt.hlines(half_amplitude, half_xmin, half_xmax, color="C2")#Afficher la durée de demi pic
    plt.scatter(spike_times,spike_y,label='spike detected',color='orange')
    plt.xlabel('Time (s)') ; plt.ylabel('Signal ({})'.format(units[-2:]))   
    
    '''Plot all the waveforms and the average waveform of a neuron'''
    # fig, subplot = plt.subplots(2,1,sharex=True) #subplots (lignes,colonnes,sharex=true pour partager le même axe x entre les plots)
    # subplot[0].set_title('All the waveforms Manip {} {}'.format(date, manip))
    # subplot[0].set_xlabel('Time (ms)'), subplot[0].set_ylabel('Signal ({})'.format(units[-2:]))
    # subplot[0].scatter(NEGSPIKES_TIMES_ms,NEGSPIKES_Y,label='negspike detected',color='red')
    # for wf in spike_wfs:
    #     subplot[0].plot(spike_wfs_times,wf,alpha=0.8)

    # subplot[1].set_title('Waveform average Manip {} {}'.format(date, manip))
    # subplot[1].set_xlabel('Time (ms)'), subplot[1].set_ylabel('Signal ({})'.format(units[-2:]))
    # subplot[1].plot(spike_wfs_times,spike_wfs_average,alpha=0.8)
    # subplot[1].errorbar(spike_wfs_times,spike_wfs_average,yerr=spike_wfs_average_sem)
    # plt.tight_layout()


     
    
    
    ''' -----------Save as excel----------'''
    '''Save the spikes waveforms of the neuron'''
    # df = pd.DataFrame(np.asarray(spike_wfs))
    # df.to_excel('{}/{}_waveforms.xlsx'.format(savedir,manip)) 

    '''Save the spike waveform average of the neuron'''
    # dfx = pd.DataFrame(np.asarray(spike_wfs_average))
    # dfx.to_excel('{}/{}_waveformsaverage.xlsx'.format(savedir,manip))
    
    '''Save the spike times of the neuron'''
    # df2 = pd.DataFrame(spike_times)
    # df2.to_excel('{}/{}_spike_times.xlsx'.format(savedir,manip))      
    
    '''Save the Half width of each spike of the neuron'''
    # data_width = { 'width' : half_width, 'amplitude' : half_amplitude, 't1' : half_xmin, 't2' : half_xmax}
    # df3 = pd.DataFrame(data_width)
    # df3.to_excel('{}/{}_half_width.xlsx'.format(savedir,manip))
    
    '''Save the full spike width of each spike of the neuron'''
    # negspike = {'Time' : NEGSPIKES_TIMES, 'Amplitude' : NEGSPIKES_Y} #Make an array Time - amplitude
    # df4 = pd.DataFrame(negspike)
    # df4.to_excel('{}/{}_test.xlsx'.format(savedir,manip))
    
    '''Save the CritA - CritB of each spike of the neuron'''
    # data_critere = { 'A' : NEGSPIKES_TIMES, 'B' : half_width}
    # df5 = pd.DataFrame(data_critere)
    # df5.to_excel('{}/{}_{}_crit.xlsx'.format(savedir, date, manip))
    '''Save Spike time - Crit A - Crit B of each spike'''
    # data_critere_spike = { 'A' : spike_times, 'B' : NEGSPIKES_TIMES, 'C' : half_width}
    # df5 = pd.DataFrame(data_critere_spike)
    # df5.to_excel('{}/{}_{}_Time_crit.xlsx'.format(savedir, date, manip))

'''Plot all the average waveform'''
# np.asarray(AVERAGE_WFS)
# AVERAGE_WFS_transposed = np.transpose(AVERAGE_WFS)
# np.asarray(AVERAGE_WFS_SEM)
# plt.figure()
# plt.title('Waveforms averages')
# plt.xlabel('Time (ms)'), plt.ylabel('Signal ({})'.format(units[-2:]))
# plt.plot(AVERAGE_WFS_transposed,alpha=0.8)
# plt.errorbar(spike_wfs_times,AVERAGE_WFS,yerr=AVERAGE_WFS_SEM)