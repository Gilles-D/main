# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 14:42:45 2024

@author: MOCAP
"""
#%% Imports
import spikeinterface as si
import spikeinterface.extractors as se 
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.comparison as sc
import spikeinterface.exporters as sexp
import spikeinterface.widgets as sw

import os
import sys
import time

import probeinterface as pi
from probeinterface.plotting import plot_probe

import warnings
warnings.simplefilter("ignore")


import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


from viziphant.statistics import plot_time_histogram
from viziphant.rasterplot import rasterplot_rates
from elephant.statistics import time_histogram
from neo.core import SpikeTrain
from quantities import s, ms
import pandas as pd

def plot_maker(sorter, we, save, sorter_name, save_path,saving_name):
    """
    Generate and save plots for an individual sorter's results.
    
    Parameters:
        sorter (spikeinterface.SortingExtractor): The sorting extractor containing the results of a spike sorter.
        we (spikeinterface.WaveformExtractor): The waveform extractor for the sorting extractor.
        save (bool): Whether to save the generated plots.
        sorter_name (str): Name of the spike sorter.
        save_path (str): Directory where the plots will be saved.
        saving_name (str): Name of the recording data.
        
    Returns:
        None
    """
    
    for unit_id in sorter.get_unit_ids():
        fig = plt.figure(figsize=(25, 13))
        gs = GridSpec(nrows=3, ncols=6)
        fig.suptitle(f'{sorter_name}\n{saving_name}\nunits {unit_id} (Total spike {sorter.get_total_num_spikes()[unit_id]})',)
        ax0 = fig.add_subplot(gs[0, 0:3])
        ax1 = fig.add_subplot(gs[0, 3:7])
        ax1.set_title('Mean firing rate during a trial')
        ax2 = fig.add_subplot(gs[1, :])
        ax2.set_title('Waveform of the unit')
        ax3 = fig.add_subplot(gs[2, 0])
        ax4 = fig.add_subplot(gs[2, 1], sharey = ax3)
        ax5 = fig.add_subplot(gs[2, 2], sharey = ax3)
        ax6 = fig.add_subplot(gs[2, 3:6])
        sw.plot_autocorrelograms(sorter, unit_ids=[unit_id], axes=ax0, bin_ms=1, window_ms=200)
        ax0.set_title('Autocorrelogram')
        current_spike_train = sorter.get_unit_spike_train(unit_id)/sorter.get_sampling_frequency()
        current_spike_train_list = []
        while len(current_spike_train) > 0: #this loop is to split the spike train into trials with correct duration in seconds
            # Find indices of elements under 9 (9 sec being the duration of the trial)
            indices = np.where(current_spike_train < 9)[0]
            if len(indices)>0:
                # Append elements to the result list
                current_spike_train_list.append(SpikeTrain(current_spike_train[indices]*s, t_stop=9))
                # Remove the appended elements from the array
                current_spike_train = np.delete(current_spike_train, indices)
                # Subtract 9 from all remaining elements
            current_spike_train -= 9
        bin_size = 100
        histogram = time_histogram(current_spike_train_list, bin_size=bin_size*ms, output='mean')
        histogram = histogram*(1000/bin_size)
        ax1.axvspan(0, 0.5, color='green', alpha=0.3)
        ax1.axvspan(1.5, 2, color='green', alpha=0.3)
        ax6.axvspan(0, 0.5, color='green', alpha=0.3)
        ax6.axvspan(1.5, 2, color='green', alpha=0.3)
        plot_time_histogram(histogram, units='s', axes=ax1)
        sw.plot_unit_waveforms_density_map(we, unit_ids=[unit_id], ax=ax2)
        template = we.get_template(unit_id=unit_id).copy()
        
        for curent_ax in [ax3, ax4, ax5]:
            max_channel = np.argmax(np.abs(template))%template.shape[1]
            template[:,max_channel] = 0
            mean_residual = np.mean(np.abs((we.get_waveforms(unit_id=unit_id)[:,:,max_channel] - we.get_template(unit_id=unit_id)[:,max_channel])), axis=0)
            curent_ax.plot(mean_residual)
            curent_ax.plot(we.get_template(unit_id=unit_id)[:,max_channel])
            curent_ax.set_title('Mean residual of the waveform for channel '+str(max_channel))
        plt.tight_layout()
        rasterplot_rates(current_spike_train_list, ax=ax6, histscale=0.1)
        if save:
            plt.savefig(fr'{save_path}\{saving_name}\we\Unit_{int(unit_id)}.pdf')
            plt.savefig(fr'{save_path}\{saving_name}\we\Unit_{int(unit_id)}.png')
            plt.close()



"""------------------Concatenation------------------"""

savename='test16'

recordings=['D:/ePhy/Intan_Data/0032/0032_01_10/0032_01_240110_151532/0032_01_240110_151532.rhd',
            'D:/ePhy/Intan_Data/0032/0032_01_10/0032_01_240110_152010/0032_01_240110_152010.rhd',
            "D:/ePhy/Intan_Data/0032/0032_01_10/0032_01_240110_152405/0032_01_240110_153405.rhd",
            "D:/ePhy/Intan_Data/0032/0032_01_10/0032_01_240110_152405/0032_01_240110_152405.rhd",
            "D:/ePhy/Intan_Data/0032/0032_01_10/0032_01_240110_154958/0032_01_240110_154958.rhd",
"D:/ePhy/Intan_Data/0032/0032_01_10/0032_01_240110_154958/0032_01_240110_155958.rhd",
"D:/ePhy/Intan_Data/0032/0032_01_10/0032_01_240110_160442/0032_01_240110_160442.rhd",
"D:/ePhy/Intan_Data/0032/0032_01_10/0032_01_240110_160442/0032_01_240110_161442.rhd",
'D:/ePhy/Intan_Data/0032/0032_01_10/0032_01_240110_162416/0032_01_240110_162416.rhd',
'D:/ePhy/Intan_Data/0032/0032_01_10/0032_01_240110_163012/0032_01_240110_163012.rhd'
            ]
recordings_list=[]

for recording_file in recordings:
    recording = se.read_intan(recording_file,stream_id='0')
    recording.annotate(is_filtered=False)
    recordings_list.append(recording)

multirecording = si.concatenate_recordings(recordings_list)
         

"""------------------Set the probe------------------"""
probe = pi.io.read_probeinterface('D:/ePhy/SI_Data/A1x16-Poly2-5mm-50s-177.json')
probe = probe.probes[0]
multirecording = multirecording.set_probe(probe)
plot_probe(probe, with_device_index=True)


"""------------------Defective sites exclusion------------------"""
sw.plot_timeseries(multirecording, channel_ids=multirecording.get_channel_ids(),time_range=[10,30])

    

"""------------------Pre Processing------------------"""
#Bandpass filter
recording_f = spre.bandpass_filter(multirecording, freq_min=300, freq_max=6000)
w = sw.plot_timeseries(recording_f,time_range=[10,30], segment_index=0)


for i in [num for num in range(300, 6000 + 1) if num % 200 == 0]:
    recording_f = spre.notch_filter(recording_f, freq=i)


#Median common ref

recording_cmr = spre.common_reference(recording_f, reference='global', operator='median', dtype='float64')
w = sw.plot_timeseries(recording_cmr,time_range=[10,30], segment_index=0)


rec_binary = recording_cmr.save(format='binary',folder=rf'D:\ePhy\SI_Data\concatenated_signals/{savename}/', n_jobs=1, progress_bar=True, chunk_duration='1s')
  
param_sorter = {
    'kilosort2_5':{
        
        }}

sorter_result = ss.run_sorter('kilosort3',recording=recording_cmr,output_folder=rf'D:\ePhy\SI_Data\spikesorting_results/{savename}/',docker_image=True,verbose=True,remove_existing_folder=True)

we = si.extract_waveforms(rec_binary, sorter_result, folder=f'D:\ePhy\SI_Data\spikesorting_results/{savename}\we')
plot_maker(sorter_result, we, save=True, sorter_name='kilosort3', save_path='D:\ePhy\SI_Data\spikesorting_results',saving_name=savename)

print("Spike sorting done")