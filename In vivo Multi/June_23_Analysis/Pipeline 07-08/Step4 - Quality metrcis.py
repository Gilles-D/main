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


#%% Parameters
spikesorting_results_folder = r'D:\ePhy\SI_Data\spikesorting_results'
session_name = '0026_05_08'



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
