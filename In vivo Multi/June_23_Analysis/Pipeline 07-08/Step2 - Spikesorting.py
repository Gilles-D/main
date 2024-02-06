# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 11:35:43 2023

@author: Gilles Delbecq

Perform spikinterface spikesorting
Sorters (and parameters) can be set in the function spike_sorting()


Inputs = concatenated signals (one or multiple sessions)
Outputs = sorting results (spikeinterface)


"""

#%% modules
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

#%% Functions

def spike_sorting(record,spikesorting_results_folder,saving_name,use_docker=True,nb_of_agreement=2,plot_sorter=True,plot_comp=True,save=True,export_to_phy=True):
    """
    Perform spike sorting using multiple sorters and compare the results.

    Parameters:
        record (spikeinterface.RecordingExtractor): The recording extractor to be sorted.
        spikesorting_results_folder (str): Directory where the results of spike sorting will be saved.
        saving_name (str): Name used for creating subdirectories for each sorter's results.
        use_docker (bool, optional): Whether to use Docker to run the spike sorters. Defaults to True.
        nb_of_agreement (int, optional): Minimum number of sorters that must agree to consider a unit as valid. Defaults to 2.
        plot_sorter (bool, optional): Whether to enable plotting of individual sorter's results. Defaults to True.
        plot_comp (bool, optional): Whether to enable plotting of comparison results. Defaults to True.
        save (bool, optional): Whether to save the plot images. Defaults to True.

    Returns:
        None
    """
    
    param_sorter = {
        'kilosort3':{
            
            },   
        # 'mountainsort4': {
        #                                 'detect_sign': -1,  # Use -1, 0, or 1, depending on the sign of the spikes in the recording
        #                                 'adjacency_radius': -1,  # Use -1 to include all channels in every neighborhood
        #                                 'freq_min': 300,  # Use None for no bandpass filtering
        #                                 'freq_max': 6000,
        #                                 'filter': True,
        #                                 'whiten': True,  # Whether to do channel whitening as part of preprocessing
        #                                 'num_workers': 1,
        #                                 'clip_size': 50,
        #                                 'detect_threshold': 3,
        #                                 'detect_interval': 10,  # Minimum number of timepoints between events detected on the same channel
        #                                 'tempdir': None
        #                             },
        #             'tridesclous': {
        #                                 'freq_min': 300.,   #'High-pass filter cutoff frequency'
        #                                 'freq_max': 6000.,#'Low-pass filter cutoff frequency'
        #                                 'detect_sign': -1,     #'Use -1 (negative) or 1 (positive) depending on the sign of the spikes in the recording',
        #                                 'detect_threshold': 5, #'Threshold for spike detection',
        #                                 'n_jobs' : 8,           #'Number of jobs (when saving ti binary) - default -1 (all cores)',
        #                                 'common_ref_removal': True,     #'remove common reference with median',
        #                             },
        #             'spykingcircus': {
        #                                 'detect_sign': -1,  #'Use -1 (negative),1 (positive) or 0 (both) depending on the sign of the spikes in the recording'
        #                                 'adjacency_radius': 100,  # Radius in um to build channel neighborhood
        #                                 'detect_threshold': 6,  # Threshold for detection
        #                                 'template_width_ms': 3,  # Template width in ms. Recommended values: 3 for in vivo - 5 for in vitro
        #                                 'filter': True, # Enable or disable filter
        #                                 'merge_spikes': True, #Enable or disable automatic mergind
        #                                 'auto_merge': 0.75, #Automatic merging threshold
        #                                 'num_workers': None, #Number of workers (if None, half of the cpu number is used)
        #                                 'whitening_max_elts': 1000,  # Max number of events per electrode for whitening
        #                                 'clustering_max_elts': 10000,  # Max number of events per electrode for clustering
        #                             }
                   }
    
    print("Spike sorting starting")

    sorter_list = []
    sorter_name_list = []
    for sorter_name, sorter_param in param_sorter.items():
        print(sorter_name)
        
        output_folder = rf'{spikesorting_results_folder}\{saving_name}\{sorter_name}'
        
        if os.path.isdir(output_folder):
            print('Sorter folder found, load from folder')
            sorter_result = se.NpzSortingExtractor.load_from_folder(rf'{output_folder}/in_container_sorting')
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
        
        if export_to_phy == True:
            
            save_folder_phy = rf'{output_folder}\phy_export'
            if os.path.exists(save_folder_phy) and os.path.isdir(save_folder_phy):
                print("Phy export already exists")
                
            else:
                print('Export to Phy...')
                #Export to phy
                sexp.export_to_phy(we, output_folder=save_folder_phy)
    
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
            sorting_agreement = se.NpzSortingExtractor.load_from_folder(f'{base_comp_folder}\sorter')
        else:
            print('multiple comparaison sorter folder not found, computing from sorter list')
            comp_multi = sc.compare_multiple_sorters(sorting_list=sorter_list,
                                                    name_list=sorter_name_list)
            comp_multi.save_to_folder(base_comp_folder)
            # del sorting_list, sorting_name_list
            sorting_agreement = comp_multi.get_agreement_sorting(minimum_agreement_count=nb_of_agreement)
            sorting_agreement._is_json_serializable = False

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
            
    

    return sorter_result,we

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
            plt.savefig(fr'{save_path}\{saving_name}\{sorter_name}\we\Unit_{int(unit_id)}.pdf')
            plt.savefig(fr'{save_path}\{saving_name}\{sorter_name}\we\Unit_{int(unit_id)}.png')
            plt.close()


#%%Parameters

#####################################################################
###################### TO CHANGE ####################################
#####################################################################
#Folder containing the folders of the session

concatenated_signals = [
# "D:/ePhy/SI_Data/concatenated_signals/0035_26_01",
# "D:/ePhy/SI_Data/concatenated_signals/0034_24_01",
# "D:/ePhy/SI_Data/concatenated_signals/0033_22_01",
"D:/ePhy/SI_Data/concatenated_signals/0032_01_10",
# "D:/ePhy/SI_Data/concatenated_signals/0031_01_10",
# "D:/ePhy/SI_Data/concatenated_signals/0030_01_09",
# "D:/ePhy/SI_Data/concatenated_signals/0030_01_11"
    ]


# saving_name="0026_02_08"
spikesorting_results_folder='D:\ePhy\SI_Data\spikesorting_results'
concatenated_files_folder = 'D:/ePhy/SI_Data/concatenated_signals'



#%%
for session in concatenated_signals:
    session_name=os.path.basename(session)
    print(session_name)
    recording = si.load_extractor(session)
    sorting = spike_sorting(recording,spikesorting_results_folder,session_name,plot_sorter=True, plot_comp=True, export_to_phy = False)


#%% Get sorter param
params = ss.get_default_sorter_params(sorter_name_or_class='kilosort3')
print("Parameters:\n", params)

desc = ss.get_sorter_params_description(sorter_name_or_class='kilosort3')
print("Descriptions:\n", desc)