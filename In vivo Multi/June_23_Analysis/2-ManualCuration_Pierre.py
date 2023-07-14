# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 18:14:52 2023

@author: Gil
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




def higher_channel_order_dict(recording_path,spikesorting_results_path, sorter_list, saving_spike_path, saving_waveform_path, saving_summary_plot, unit_base_nb=0, number_of_channel_for_comp=2):
    # Initialize dictionaries
    higher_channel_dict = {}
    sorter_dict = {}

    # Construct recording path based on 'mouse' and 'recording_delay'
    recording_path = recording_path

    # Change current working directory to recording_path
    os.chdir(recording_path)

    # Load binary recording 'multirecording' from the 'concatenated_recording' folder
    multirecording = si.BinaryFolderRecording(recording_path)
    multirecording.annotate(is_filtered=True)

    # Initialize waveform extractor lists
    we_list = []
    we_name_list = []

    # Loop over each sorter in sorter_list
    for sorter in sorter_list:
        print(sorter)
        # Assign abbreviated names to sorters
        if sorter == 'comp_mult_2_tridesclous_spykingcircus_mountainsort4':
            sorter_mini = 'comp'
        elif sorter == 'herdingspikes':
            sorter_mini = 'her'
        elif sorter == 'mountainsort4':
            sorter_mini = 'moun'
        elif sorter == 'tridesclous':
            sorter_mini = 'tdc'
        else:
            sorter_mini = sorter

        # Construct spike folder path based on 'mouse', 'delay', and 'sorter'
        spike_folder = rf'{spikesorting_results_path}/{sorter}'
        
        if sorter_mini == 'comp':
            # Load sorting results for 'comp' sorter
            print(f'{spike_folder}\sorter')
            sorter_result = ss.NpzSortingExtractor.load_from_folder(f'{spike_folder}\sorter')
        else:
            # Load sorting results for other sorters
            print(f'{spike_folder}\sorter\in_container_sorting')
            sorter_result = ss.NpzSortingExtractor.load_from_folder(f'{spike_folder}\sorter\in_container_sorting')

        # Load waveform extractor ('we') from the 'we' folder
        we = si.WaveformExtractor.load_from_folder(f'{spike_folder}\we', sorting=sorter_result)

        # Add waveform extractor and sorter name to the respective lists
        we_list.append(we)
        we_name_list.append(sorter_mini)

        # Loop over each unit in the sorter results
        for unit in sorter_result.get_unit_ids():
            waveform = pd.DataFrame(we.get_template(unit))
            waveform = waveform.abs()

            higher_channel_list = []

            # Extract channels with highest amplitudes until the desired number is reached
            while len(waveform.columns) > (16 - number_of_channel_for_comp):
                higher_channel = waveform.max().idxmax()
                waveform = waveform.drop(higher_channel, axis=1)
                higher_channel_list.append(int(higher_channel))

            # Store higher channel list in higher_channel_dict
            higher_channel_dict[f'{sorter_mini}_{unit}'] = higher_channel_list

            # Store sorter result and waveform extractor in sorter_dict
            if sorter_mini not in sorter_dict.keys():
                sorter_dict[sorter_mini] = {'sorter': sorter_result, 'we': we}

    # Call frozenset_dict_maker to create an immutable dictionary from higher_channel_dict
    higher_channel_order_dict = frozenset_dict_maker(higher_channel_dict)

    # Call unit_high_channel_ploting with appropriate arguments
    unit_base_nb = unit_high_channel_ploting(higher_channel_order_dict, sorter_dict, mouse, delay, saving_spike_path, saving_waveform_path, saving_summary_plot, unit_base_nb)



"""
Parameters
"""

recording_name="0012_03_07_allfiles_allchan"
sorter_list = ['comp_mult_2_tridesclous_spykingcircus_mountainsort4', 'mountainsort4','spykingcircus', 'tridesclous']



# Saving Folder path
concatenated_signals_path=r"D:\Seafile\Seafile\Data\ePhy\2_SI_data\concatenated_signals"
recording_path = rf'{concatenated_signals_path}/{recording_name}'
spikesorting_results_path=rf"D:\Seafile\Seafile\Data\ePhy\2_SI_data\spikesorting_results/{recording_name}"


saving_spike_path = rf'{spikesorting_results_path}/{recording_name}/spikes'
saving_waveform_path = rf'{spikesorting_results_path}/{recording_name}/waveforms'
saving_summary_plot = rf'{spikesorting_results_path}/{recording_name}/summary_plots'


higher_channel_order_dict(recording_path,spikesorting_results_path,sorter_list,saving_spike_path,saving_waveform_path,saving_summary_plot)
