# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:59:35 2023

@author: MOCAP
"""

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


def concatenate_preprocessing(recordings,saving_dir,saving_name,probe_path,excluded_sites,freq_min=300,freq_max=6000,Plotting=True):
    if os.path.isdir(rf'{saving_dir}/{saving_name}/'):
        print('Concatenated file already exists')
        rec_binary = si.load_extractor(rf'{saving_dir}/{saving_name}/')
    
    else:
        print('Concatenating...')
        
        """------------------Concatenation------------------"""
        recordings_list=[]
        for recording_file in recordings:
            recording = se.read_intan(recording_file,stream_id='0')
            recording.annotate(is_filtered=False)
            recordings_list.append(recording)
        
        multirecording = si.concatenate_recordings(recordings_list)

        """------------------Set the probe------------------"""
        probe = pi.io.read_probeinterface(probe_path)
        probe = probe.probes[0]
        multirecording = multirecording.set_probe(probe)
        if Plotting==True:
            plot_probe(probe, with_channel_index=True, with_device_index=True)
        
        
        
        """------------------Defective sites exclusion------------------"""
        if Plotting==True:
            sw.plot_timeseries(multirecording, channel_ids=multirecording.get_channel_ids(),time_range=[10,13])
        
        multirecording.set_channel_groups(1, excluded_sites)
        multirecording = multirecording.split_by('group')[0]
        if Plotting==True:
            sw.plot_timeseries(multirecording, channel_ids=multirecording.get_channel_ids(),time_range=[10,13])
        
        
        """------------------Pre Processing------------------"""
        #Bandpass filter
        recording_f = spre.bandpass_filter(multirecording, freq_min=freq_min, freq_max=freq_max)
        if Plotting==True:
            w = sw.plot_timeseries(recording_f,time_range=[10,13], segment_index=0)
        
        
        #Median common ref
        recording_cmr = spre.common_reference(recording_f, reference='global', operator='median')
        if Plotting==True:
            w = sw.plot_timeseries(recording_cmr,time_range=[10,13], segment_index=0)
        
        rec_binary = recording_cmr.save(folder=rf'{saving_dir}/{saving_name}/', n_jobs=1, progress_bar=True, chunk_duration='1s')
    
    #TODO Save infos : excluded chan, probe, frequencies for filtering
    trial_time_index_df=pd.DataFrame({'concatenated_time':multirecording.get_times()})


    with open(rf'{saving_dir}/{saving_name}/concatenated_recording_trial_time_index_df.pickle', 'wb') as file:
        pickle.dump(trial_time_index_df, file, protocol=pickle.HIGHEST_PROTOCOL)   

    return rec_binary


def spike_sorting(record,spikesorting_results_folder,saving_name,use_docker=True,nb_of_agreement=2,plot_sorter=True,plot_comp=True,save=True):
    param_sorter = {
                    'tridesclous': {
                                        'freq_min': 300.,   #'High-pass filter cutoff frequency'
                                        'freq_max': 6000.,#'Low-pass filter cutoff frequency'
                                        'detect_sign': -1,     #'Use -1 (negative) or 1 (positive) depending on the sign of the spikes in the recording',
                                        'detect_threshold': 5, #'Threshold for spike detection',
                                        'n_jobs' : 8,           #'Number of jobs (when saving ti binary) - default -1 (all cores)',
                                        'common_ref_removal': True,     #'remove common reference with median',
                                    },
                    'spykingcircus': {
                                        'detect_sign': -1,  #'Use -1 (negative),1 (positive) or 0 (both) depending on the sign of the spikes in the recording'
                                        'adjacency_radius': 100,  # Radius in um to build channel neighborhood
                                        'detect_threshold': 6,  # Threshold for detection
                                        'template_width_ms': 3,  # Template width in ms. Recommended values: 3 for in vivo - 5 for in vitro
                                        'filter': True, # Enable or disable filter
                                        'merge_spikes': True, #Enable or disable automatic mergind
                                        'auto_merge': 0.75, #Automatic merging threshold
                                        'num_workers': None, #Number of workers (if None, half of the cpu number is used)
                                        'whitening_max_elts': 1000,  # Max number of events per electrode for whitening
                                        'clustering_max_elts': 10000,  # Max number of events per electrode for clustering
                                    },
                    'mountainsort4': {
                                        'detect_sign': -1,  # Use -1, 0, or 1, depending on the sign of the spikes in the recording
                                        'adjacency_radius': -1,  # Use -1 to include all channels in every neighborhood
                                        'freq_min': 300,  # Use None for no bandpass filtering
                                        'freq_max': 6000,
                                        'filter': True,
                                        'whiten': True,  # Whether to do channel whitening as part of preprocessing
                                        'num_workers': 1,
                                        'clip_size': 50,
                                        'detect_threshold': 3,
                                        'detect_interval': 10,  # Minimum number of timepoints between events detected on the same channel
                                        'tempdir': None
                                    }
                 }
    
    print("Spike sorting starting")

    sorter_list = []
    sorter_name_list = []
    for sorter_name, sorter_param in param_sorter.items():
        print(sorter_name)
        
        output_folder = rf'{spikesorting_results_folder}\{saving_name}\{sorter_name}'
        
        if os.path.isdir(output_folder):
            print('Sorter folder found, load from folder')
            sorter_result = ss.NpzSortingExtractor.load_from_folder(rf'{output_folder}/in_container_sorting')
        else:
            sorter_result = ss.run_sorter(sorter_name,recording=record,output_folder=output_folder,docker_image=True,verbose=False,**sorter_param)
        
              
        sorter_list.append(sorter_result)
        sorter_name_list.append(sorter_name)
    

        #save the sorter params
        with open(f'{output_folder}\param_dict.pkl', 'wb') as f:
            pickle.dump(sorter_param, f)
        if os.path.isdir(f'{output_folder}\we'):
            print('Waveform folder found, load from folder')
            we = si.WaveformExtractor.load_from_folder(f'{output_folder}\we', sorting=sorter_result)
        else:
            we = si.extract_waveforms(record, sorter_result, folder=f'{output_folder}\we')
    
        if plot_sorter:
            print('Plot sorting summary in progress')
            plot_maker(sorter_result, we, save, sorter_name, spikesorting_results_folder,saving_name)
            print('Plot sorting summary finished')
        print('================================')
    
    print("Spike sorting done")
    
    
    
    if len(sorter_list) > 1 and nb_of_agreement != 0:
        ############################
        # Sorter outup comparaison #
        base_comp_folder = rf'{spikesorting_results_folder}\{saving_name}'
        comp_multi_name = f'comp_mult_{nb_of_agreement}'
        
        for sorter_name in sorter_name_list:
            comp_multi_name += f'_{sorter_name}'
        base_comp_folder = f'{base_comp_folder}\{comp_multi_name}'

        if os.path.isdir(f'{base_comp_folder}\sorter'):
            print('multiple comparaison sorter folder found, load from folder')
            sorting_agreement = ss.NpzSortingExtractor.load_from_folder(f'{base_comp_folder}\sorter')
        else:
            print('multiple comparaison sorter folder not found, computing from sorter list')
            comp_multi = sc.compare_multiple_sorters(sorting_list=sorter_list,
                                                    name_list=sorter_name_list)
            comp_multi.save_to_folder(base_comp_folder)
            # del sorting_list, sorting_name_list
            sorting_agreement = comp_multi.get_agreement_sorting(minimum_agreement_count=nb_of_agreement)
            sorting_agreement.save_to_folder(f'{base_comp_folder}\sorter')
        
        
        try:
            we = si.extract_waveforms(record, sorting_agreement, folder=f'{base_comp_folder}\we')
        
        
        except FileExistsError:
            print('multiple comparaison waveform folder found, load from folder')
            we = si.WaveformExtractor.load_from_folder(f'{base_comp_folder}\we', sorting=sorting_agreement)
        
        
        if plot_comp:
            print('Plot multiple comparaison summary in progress')
            plot_maker(sorting_agreement, we, save, comp_multi_name, spikesorting_results_folder,saving_name)
            print('Plot multiple comparaison summary finished\n')
    
    
    
    
    
    
    return
    

def load_concatenated_recordings(saving_dir,saving_name):
    return si.load_extractor(rf'{saving_dir}/{saving_name}/')



def plot_maker(sorter, we, save, sorter_name, save_path,saving_name):
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
            plt.savefig(fr'{save_path}\{saving_name}\{sorter_name}\we\Unit_{int(unit_id)}.pdf')
            plt.close()




#%%Parameters

probe_path=r'D:/ePhy/SI_Data/A1x16-Poly2-5mm-50s-177.json'   #INTAN Optrode
# probe_path = 'D:/ePhy/SI_Data/Buzsaki16.json'              #INTAN Buzsaki16

# Saving Folder path
saving_dir=r"D:/ePhy/SI_Data/concatenated_signals"
saving_name="0012_03_07_allfiles_allchan"

excluded_sites = []


recordings=["D:/ePhy/Intan_Data/0012/07_03/0012_07_03_File_06.rhd",
"D:/ePhy/Intan_Data/0012/07_03/0012_07_03_File_01.rhd",
"D:/ePhy/Intan_Data/0012/07_03/0012_07_03_File_02.rhd",
"D:/ePhy/Intan_Data/0012/07_03/0012_07_03_File_03.rhd",
"D:/ePhy/Intan_Data/0012/07_03/0012_07_03_File_04.rhd",
"D:/ePhy/Intan_Data/0012/07_03/0012_07_03_File_05.rhd"
    ]

spikesorting_results_folder='D:\ePhy\SI_Data\spikesorting_results'


#%%
recording = concatenate_preprocessing(recordings,saving_dir,saving_name,probe_path,excluded_sites,Plotting=True)

spike_sorting(recording,spikesorting_results_folder,saving_name,plot_sorter=True, plot_comp=True)
