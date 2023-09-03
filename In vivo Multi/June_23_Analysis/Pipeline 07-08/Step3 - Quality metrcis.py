# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 10:07:55 2023

@author: MOCAP
"""

#%% Imports and functions
import spikeinterface as si
import spikeinterface.extractors as se 
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.comparison as sc
import spikeinterface.exporters as sexp
import spikeinterface.widgets as sw
from spikeinterface.curation import MergeUnitsSorting, get_potential_auto_merge

import os

import probeinterface as pi
from probeinterface.plotting import plot_probe

import warnings
warnings.simplefilter("ignore")


import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from matplotlib.widgets import Button

from viziphant.statistics import plot_time_histogram
from viziphant.rasterplot import rasterplot_rates
from elephant.statistics import time_histogram
from neo.core import SpikeTrain
from quantities import s, ms

import pandas as pd
import math


def list_curated_units(directory):
    units = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            unit_id = file_name.split('_')[1].split('(')[0].split(' ')[0]
            unit_org_id = file_name.split('_')[3]
            unit_sorter = file_name.split('_')[-1].split(')')[0]
            units.append((unit_id,unit_org_id,unit_sorter))
    return units

def get_similarity_couples(similarity,similarity_threshold):
    row_indices, col_indices = np.where(similarity > similarity_threshold)
    # Utiliser un ensemble pour stocker les couples d'unités uniques
    unique_couples = set()

    # Parcourir les indices et ajouter les couples d'unités uniques à l'ensemble
    for row_idx, col_idx in zip(row_indices, col_indices):
        if row_idx != col_idx:
            unique_couples.add((min(row_idx, col_idx), max(row_idx, col_idx)))

    # Convertir l'ensemble en liste de couples d'unités
    couples_unites = list(unique_couples)
    
    return couples_unites


def plot_maker(sorter, we,unit_list):
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
    
    for unit_id in unit_list:
        fig = plt.figure(figsize=(25, 13))
        gs = GridSpec(nrows=3, ncols=6)
        fig.suptitle(f'{unit_id} (Total spike {sorter.get_total_num_spikes()[unit_id]})',)
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




#%% Parameters
session_name = '0026_10_08'
sorter_name='kilosort3'



spikesorting_results_folder = r'D:\ePhy\SI_Data\spikesorting_results'
sorter_folder = rf'{spikesorting_results_folder}/{session_name}/{sorter_name}'




"""
https://spikeinterface.readthedocs.io/en/latest/modules/qualitymetrics.html

Criterium to exclude units :
    
    (- refractory period < 0.5% (1ms))
    - minimum frequency < 0.1 Hz
    - presence ratio > 90 (eliminate artefacts?)
    - ISI violation ratio > 5
    - L ratio > 10 ?
    
"""



max_isi = 5
min_frequency = 0.1
min_presence = 0.9
max_l_ratio = 10

similarity_threshold = 0.9


#%% One sorter auto-curation
"""
Loading
"""
sorter_result = ss.NpzSortingExtractor.load_from_folder(rf'{sorter_folder}/in_container_sorting')
we = si.WaveformExtractor.load_from_folder(f'{sorter_folder}\we')
similarity = np.load(rf"{sorter_folder}\we\similarity\similarity.npy")

units_location = spost.compute_unit_locations(we)


"""
Computing metrics
"""
# qm_params = sqm.get_default_qm_params()
# print(qm_params)
try:
    quality_metrics = sqm.compute_quality_metrics(we, load_if_exists=True)
except:
    quality_metrics = sqm.compute_quality_metrics(we, load_if_exists=False)
spike_id_list = quality_metrics.index

"""
Filtering with criteria
"""
# Filter the units with good quality metrics, according to the selected parameters
crit_ISI = quality_metrics['isi_violations_ratio'] < max_isi
crit_frequency = quality_metrics['firing_rate'] > min_frequency
crit_presence = quality_metrics['presence_ratio'] > min_presence
crit_l_ratio = quality_metrics['l_ratio'] < max_l_ratio

selected_quality_metrics = quality_metrics[crit_ISI & crit_frequency & crit_presence & crit_l_ratio]
selected_spike_id_list = selected_quality_metrics.index

spikes_not_passing_quality_metrics = list(set(spike_id_list) - set(selected_spike_id_list))

print(rf"{spikes_not_passing_quality_metrics} removed")

#%% Similarity computing
# Detect the units with high similarity and save them

similarity_couples = get_similarity_couples(similarity,similarity_threshold)
similarity_couples_indexed = []

for indices_tuple in similarity_couples:
    valeurs_tuple = [spike_id_list[idx] for idx in indices_tuple]
    
    if valeurs_tuple[0] in selected_spike_id_list and valeurs_tuple[1] in selected_spike_id_list :
        similarity_couples_indexed.append(valeurs_tuple)

print(rf"Similarity couples to check manually {similarity_couples_indexed}")

print("Next step = manually curate with phy")
print("Select what units to merge and write them in a list of lists")

for couple in similarity_couples_indexed:
    plot_maker(sorter_result,we,couple)

#%% AFTER MANUAL CURATION

# Next step = manually curate with phy
# Select what units to merge and write them in a list of lists

"""
# Calcul des distances entre chaque paire de points
num_points = units_location.shape[0]
distances = np.zeros((num_points, num_points))  # Matrice pour stocker les distances

for i in range(num_points):
    for j in range(num_points):
        distances[i, j] = np.sqrt((units_location[i, 0] - units_location[j, 0])**2 + (units_location[i, 1] - units_location[j, 1])**2)

print("Matrice des distances entre les points :\n", distances)
"""


# for units in similarity_couples_indexed:
#     sw.plot_unit_locations(we,unit_ids=[units[0], units[1]])

# for unit in selected_spike_id_list:
#     template = we.get_template(unit_id=unit, mode='median')
#     plt.figure()
#     plt.plot(template)
#     plt.title(rf"Unit # {unit}")
    
    
    

units_to_merge = [[47, 55], [57, 58,59, 69], [18, 43], [26, 46]
                  
                            
]

units_to_remove = [62, 63];

# definitive curated units list
clean_sorting = MergeUnitsSorting(sorter_result,units_to_merge).remove_units(spikes_not_passing_quality_metrics).remove_units(units_to_remove)

# save the final curated spikesorting results
save_path = rf"{sorter_folder}\curated"
clean_sorting.save_to_folder(save_path)

#save units_to_merge and units_toremove (and units pre-fitlered by metrics)
curation_infos = {
    'units_merged' : units_to_merge,
    'units_removed' : units_to_remove,
    'similarity' : similarity_couples_indexed,
    'not_passing_quality_metrics' : spikes_not_passing_quality_metrics
    
    }

pickle.dump(curation_infos, open(rf"{sorter_folder}\curated\curated_infos.pickle", "wb"))

#%%
#TODO : export to phy


#%% Script


# #List all curated units

# curated_units = list_curated_units(rf'{spikesorting_results_folder}/{session_name}/spikes')

# post_curated_units = []

# for index,unit in enumerate(curated_units):
#     sorter_name = unit[-1]
#     if sorter_name == 'comp':
#         sorter_folder = rf'{spikesorting_results_folder}/{session_name}/comp_mult_2_kilosort3_mountainsort4_tridesclous'
#         sorter_result = ss.NpzSortingExtractor.load_from_folder(rf'{sorter_folder}/sorter')
#         # we = si.WaveformExtractor.load_from_folder(f'{sorter_folder}\we', sorting=sorter_result)
#         select_unit = sorter_result.select_units(np.array([int(unit[1])]))
#         we = si.WaveformExtractor.load_from_folder(f'{sorter_folder}\we', sorting=select_unit)
        
#     elif sorter_name == 'tdc':
#         sorter_folder = rf'{spikesorting_results_folder}/{session_name}/tridesclous'
#         sorter_result = ss.NpzSortingExtractor.load_from_folder(rf'{sorter_folder}/in_container_sorting')
#         # we = si.WaveformExtractor.load_from_folder(f'{sorter_folder}\we', sorting=sorter_result)
#         select_unit = sorter_result.select_units(np.array([int(unit[1])]))
#         we = si.WaveformExtractor.load_from_folder(f'{sorter_folder}\we', sorting=select_unit)
#     elif sorter_name == 'moun':
#         sorter_folder = rf'{spikesorting_results_folder}/{session_name}/mountainsort4'
#         sorter_result = ss.NpzSortingExtractor.load_from_folder(rf'{sorter_folder}/in_container_sorting')
#         # we = si.WaveformExtractor.load_from_folder(f'{sorter_folder}\we', sorting=sorter_result)
#         select_unit = sorter_result.select_units(np.array([int(unit[1])]))
#         we = si.WaveformExtractor.load_from_folder(f'{sorter_folder}\we', sorting=select_unit)
#     else:
#         sorter_folder = rf'{spikesorting_results_folder}/{session_name}/{sorter_name}'
#         sorter_result = ss.NpzSortingExtractor.load_from_folder(rf'{sorter_folder}/in_container_sorting')
#         # we = si.WaveformExtractor.load_from_folder(f'{sorter_folder}\we', sorting=sorter_result)
#         select_unit = sorter_result.select_units(np.array([int(unit[1])]))
#         we = si.WaveformExtractor.load_from_folder(f'{sorter_folder}\we', sorting=select_unit)

#     ISI_violation_ratio,ISI_violation_count = sqm.compute_isi_violations(we, isi_threshold_ms=1.0)
    

    
    
#     quality_metrics = sqm.compute_quality_metrics(we)
#     num_spikes = int(quality_metrics['num_spikes'])
    
#     refrac_period_violation = sqm.compute_refrac_period_violations(we)
#     refrac_period_violation = list(refrac_period_violation[1].values())[0]/num_spikes*100
    
#     print(rf' Unit {unit[0]} ISI violation rate = {ISI_violation_ratio[int(unit[1])]}')
#     print(rf' Refractory period violation = {refrac_period_violation} %')
    
#     post_curated_units.append(select_unit)
    
#     # output_file = rf'{spikesorting_results_folder}/{session_name}/post_curated_spikes.h5'
#     # post_curated_units
#     # se.MultiSortingExtractor()
    
#%%
    # pc = spost.compute_principal_components(we, load_if_exists=True, n_components=3, mode='by_channel_local')
    # sw.plot_principal_component(we)
    
    # keep_mask = (metrics['snr'] > 7.5) & (metrics['isi_violations_ratio'] < 0.2) & (metrics['nn_hit_rate'] > 0.90)
    # print(keep_mask)
    
    # keep_unit_ids = keep_mask[keep_mask].index.values
    # print(keep_unit_ids)
    
    # sw.plot_autocorrelograms(sorter_result, unit_ids=[np.array([int(unit[1])])], bin_ms=1, window_ms=100)
    
    # sw.plot_quality_metrics(we)
    # plt.plot(spost.compute_isi_histograms(we))

