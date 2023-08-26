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


#%% Parameters
session_name = '0026_29_07'
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
quality_metrics = sqm.compute_quality_metrics(we, load_if_exists=True)
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

#%% Similarity computing
# Detect the units with high similarity and save them

similarity_couples = get_similarity_couples(similarity,similarity_threshold)
similarity_couples_indexed = []

for indices_tuple in similarity_couples:
    valeurs_tuple = [spike_id_list[idx] for idx in indices_tuple]
    
    if valeurs_tuple[0] in selected_spike_id_list and valeurs_tuple[1] in selected_spike_id_list :
        similarity_couples_indexed.append(valeurs_tuple)

print(rf"Similarity couples to check manually {similarity_couples_indexed}")


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
    
    
    
    

units_to_merge = [[8, 20],[12, 26],[24, 32], [25, 33], [33, 42], [32, 41], [41, 50], [31, 40], [28, 36], [42, 51],
                  
                  
                              
]

units_to_remove = [1,3,14,22,34,39,43,47,49,]

# definitive curated units list
clean_sorting = MergeUnitsSorting(sorter_result,units_to_merge).remove_units(units_to_remove)


# save the final curated spikesorting results
save_path = rf"{sorter_folder}\curated"
clean_sorting.save_to_folder(save_path)

#TODO : save units_to_merge and units_toremove (and units pre-fitlered by metrics)




#%% Script


#List all curated units

curated_units = list_curated_units(rf'{spikesorting_results_folder}/{session_name}/spikes')

post_curated_units = []

for index,unit in enumerate(curated_units):
    sorter_name = unit[-1]
    if sorter_name == 'comp':
        sorter_folder = rf'{spikesorting_results_folder}/{session_name}/comp_mult_2_kilosort3_mountainsort4_tridesclous'
        sorter_result = ss.NpzSortingExtractor.load_from_folder(rf'{sorter_folder}/sorter')
        # we = si.WaveformExtractor.load_from_folder(f'{sorter_folder}\we', sorting=sorter_result)
        select_unit = sorter_result.select_units(np.array([int(unit[1])]))
        we = si.WaveformExtractor.load_from_folder(f'{sorter_folder}\we', sorting=select_unit)
        
    elif sorter_name == 'tdc':
        sorter_folder = rf'{spikesorting_results_folder}/{session_name}/tridesclous'
        sorter_result = ss.NpzSortingExtractor.load_from_folder(rf'{sorter_folder}/in_container_sorting')
        # we = si.WaveformExtractor.load_from_folder(f'{sorter_folder}\we', sorting=sorter_result)
        select_unit = sorter_result.select_units(np.array([int(unit[1])]))
        we = si.WaveformExtractor.load_from_folder(f'{sorter_folder}\we', sorting=select_unit)
    elif sorter_name == 'moun':
        sorter_folder = rf'{spikesorting_results_folder}/{session_name}/mountainsort4'
        sorter_result = ss.NpzSortingExtractor.load_from_folder(rf'{sorter_folder}/in_container_sorting')
        # we = si.WaveformExtractor.load_from_folder(f'{sorter_folder}\we', sorting=sorter_result)
        select_unit = sorter_result.select_units(np.array([int(unit[1])]))
        we = si.WaveformExtractor.load_from_folder(f'{sorter_folder}\we', sorting=select_unit)
    else:
        sorter_folder = rf'{spikesorting_results_folder}/{session_name}/{sorter_name}'
        sorter_result = ss.NpzSortingExtractor.load_from_folder(rf'{sorter_folder}/in_container_sorting')
        # we = si.WaveformExtractor.load_from_folder(f'{sorter_folder}\we', sorting=sorter_result)
        select_unit = sorter_result.select_units(np.array([int(unit[1])]))
        we = si.WaveformExtractor.load_from_folder(f'{sorter_folder}\we', sorting=select_unit)

    ISI_violation_ratio,ISI_violation_count = sqm.compute_isi_violations(we, isi_threshold_ms=1.0)
    

    
    
    quality_metrics = sqm.compute_quality_metrics(we)
    num_spikes = int(quality_metrics['num_spikes'])
    
    refrac_period_violation = sqm.compute_refrac_period_violations(we)
    refrac_period_violation = list(refrac_period_violation[1].values())[0]/num_spikes*100
    
    print(rf' Unit {unit[0]} ISI violation rate = {ISI_violation_ratio[int(unit[1])]}')
    print(rf' Refractory period violation = {refrac_period_violation} %')
    
    post_curated_units.append(select_unit)
    
    # output_file = rf'{spikesorting_results_folder}/{session_name}/post_curated_spikes.h5'
    # post_curated_units
    # se.MultiSortingExtractor()
    
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

