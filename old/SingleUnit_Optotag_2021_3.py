# -*- coding: utf-8 -*-
"""
Created on Thu May  6 15:42:38 2021

@author: Gilles.DELBECQ
"""

"""
lit l'ensemble des smr
détermine valeur de la baseline
détecte tous les spikes à +xmv de la baseline = spiketimes
extrait les stimulation = stimtimes
eventplot
psth

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
data_dir = r'C:\Users\Gilles.DELBECQ\Desktop\single unit\traces\21-05-21\smr'


#File loading list loop
List_File_paths = []
for r, d, f in os.walk(data_dir):
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
          

JITTER=[]            
names=[]

for file_path in List_File_paths:
    '''--------Initialization--------'''

    #File identification
    exp = os.path.splitext(os.path.basename(file_path))[0]
    names.append(exp)
    print(exp)
    
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
    stim_idx_corrected=stim_times*sampling_rate
    
    #Params for spike detection
    noise_window=10
    # threshold = 0.25
    distance = 50
    negdistance = 30
    fenetre = (100/1000)*sampling_rate
    RASTER=[]
    n=0
    
    
    """
    détermine valeur de la baseline
    prendre les 10 premières secondes : mean value ?
    
    """
    
    noise = signal[0:int(noise_window*sampling_rate)]
    threshold = np.median(noise)+6*np.std(noise)
    
    
    """
        détecte tous les spikes à +xmv de la baseline = spiketimes
    """
    
    '''Positive Spikes P0'''
    #Detect the spike indexes
    spike_idx, _ = sp.find_peaks(signal,height=threshold,distance=distance)
    #Convert to spike times
    spike_times = spike_idx*1./sampling_rate
    #Get spikes peak 
    spike_y = signal[spike_idx]
    

    
    
    FIRST_SPIKES=[] 
    FIRST_SPIKES_TIMES=[] 
    for idx in stim_idx_corrected:
        # print(idx)
        a=0
        for s_idx in spike_idx:
            if s_idx > idx and s_idx <= idx+fenetre and a==0:
                spike_raster = s_idx - idx
                RASTER.append([spike_raster*1000/sampling_rate])
                FIRST_SPIKES.append(s_idx)
                FIRST_SPIKES_TIMES.append(s_idx/sampling_rate)
                a=1
    
    '''------- Plots -------- '''
    '''Plot figures spike detection'''
    # Do a figure to show spike detection result
    plt.figure()
    plt.plot(time_vector, signal, label='Data')
    plt.title('{}'.format(exp))
    plt.scatter(spike_times,spike_y,label='spike detected',color='orange')
    plt.xlabel('Time (s)') ; plt.ylabel('Signal ({})'.format(units[-2:]))
    plt.axhline(threshold)
    for i in stim_times:
        plt.axvline(i, color='red', alpha=0.05)
    for i in FIRST_SPIKES_TIMES:
        plt.axvline(i, color='blue', alpha=0.1)
        
    plt.savefig("{}\Spike Detection {}.pdf".format(data_dir,exp))
    
    
    fig, subplot = plt.subplots(2,1,sharex=True) #subplots (lignes,colonnes,sharex=true pour partager le même axe x entre les plots)
    subplot[0].set_title('RasterPlot {}'.format(exp))
    subplot[0].set_ylabel("Nombre d'événements")
    subplot[0].eventplot(RASTER, lineoffsets = 1, linelengths = 0.5, colors = 'black')
    subplot[0].axvspan(0, 2, alpha=0.1, color='blue')
    subplot[0].set_xlim(-0,25)

    width_bins = float(0.02)
    n_bins = math.ceil((max(RASTER)[0] - min(RASTER)[0])/width_bins)
    subplot[1].set_title("Peri Stimulus Histogramme {}. Bins = {}".format(exp, width_bins))
    subplot[1].set_xlabel('Time (ms)'), subplot[0].set_ylabel("Nombre d'événements")
    subplot[1].hist(np.array(RASTER), bins=n_bins, color='black')
    subplot[1].axvspan(0, 2, alpha=0.1, color='blue')
    subplot[1].set_xlim(-0,25)
    
    plt.savefig("{}\Raster+PSTH {}.pdf".format(data_dir,exp))
    
    
    '''Extraction des waveform'''  
    spike_wfs = extract_spike_waveform(signal, FIRST_SPIKES,left_width=25,right_width=25)
    spike_wfs_indexes = np.arange(-25,25,1)
    spike_wfs_times = spike_wfs_indexes*1./sampling_rate*1000 #in ms
  
    '''Waveform average of the spikes of a neuron'''
    spike_wfs_average = np.average(spike_wfs, axis=0)
    spike_wfs_average_sem = stats.sem(spike_wfs, ddof=len(spike_wfs)-2)
    
    '''Plot all the waveforms and the average waveform of a neuron'''
    fig, subplot = plt.subplots(2,1,sharex=True) #subplots (lignes,colonnes,sharex=true pour partager le même axe x entre les plots)
    subplot[0].set_title('All the waveforms {}'.format(exp))
    subplot[0].set_xlabel('Time (ms)'), subplot[0].set_ylabel('Signal ({})'.format(units[-2:]))
    for wf in spike_wfs:
        subplot[0].plot(spike_wfs_times,wf,alpha=0.8)

    subplot[1].set_title('Waveform average (n = {})'.format(len(spike_wfs)))
    subplot[1].set_xlabel('Time (ms)'), subplot[1].set_ylabel('Signal ({})'.format(units[-2:]))
    subplot[1].plot(spike_wfs_times,spike_wfs_average,alpha=0.8)
    subplot[1].errorbar(spike_wfs_times,spike_wfs_average,yerr=spike_wfs_average_sem)
    # subplot[1].text(0.5, -3.1, 'n = {}'.format(len(spike_wfs)), fontsize=10)
    # subplot[1].text(0.5, -2.8, 'Error : {}%'.format(100-(len(RASTER)/len(stim_idx)*100)), fontsize=10)
    # subplot[1].text(0.5, -2.5, 'Jitter : {}ms +/- {}ms'.format(round(np.mean(np.ravel(RASTER)),3),round(np.std(np.ravel(RASTER)),3)), fontsize=10)
    plt.tight_layout()
    plt.savefig("{}\Waveform {}.pdf".format(data_dir,exp))
    
    JITTER.append((np.mean(np.ravel(RASTER)),np.std(np.ravel(RASTER)), 100-(len(RASTER)/len(stim_idx)*100)))
df = pd.DataFrame(JITTER)
df = pd.DataFrame(JITTER,index=names,columns=['Delay', 'Jitter (std)', 'Failure (%)'])
df.to_excel('{}/datas.xlsx'.format(data_dir))