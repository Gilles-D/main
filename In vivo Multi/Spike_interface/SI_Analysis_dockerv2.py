# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 11:31:45 2023

@author: Gilles.DELBECQ
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

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# import spikeinterface_gui as sigui

import warnings
warnings.simplefilter("ignore")

import docker

#%% Section 1

"""
---------------------------PARAMETERS---------------------------

"""

use_docker = True

# Working folder path
working_dir=r'D:\ePhy\SI_Data'

subject_name="578"
recording_name='578_12-04_baseline_'


sorting_saving_dir=rf'{working_dir}/{subject_name}/sorting_output/{recording_name}'

# param_sorter=['tridesclous', 'mountainsort4', 'spykingcircus', 'waveclus', 'ironclust','herdingspikes']


param_sorter = {
    # Parameters from all sorters can be retrieved with these functions:
    # params = ss.get_default_sorter_params('spykingcircus')
    # print("Parameters:\n", params)
    # desc = ss.get_sorter_params_description('spykingcircus')
    # print("Descriptions:\n", desc)
             
    
    'kilosort3': {
                                    'detect_threshold': 6,
                                    'projection_threshold': [9, 9],
                                    'preclust_threshold': 8,
                                    'car': True,
                                    'minFR': 0.2,
                                    'minfr_goodchannels': 0.2,
                                    'nblocks': 5,
                                    'sig': 20,
                                    'freq_min': 300,
                                    'sigmaMask': 30,
                                    'nPCs': 3,
                                    'ntbuff': 64,
                                    'nfilt_factor': 4,
                                    'do_correction': True,
                                    'NT': None,
                                    'wave_length': 61,
                                    'keep_good_only': False,
                                },
                'mountainsort4': {
                                    'detect_sign': -1,  # Use -1, 0, or 1, depending on the sign of the spikes in the recording
                                    'adjacency_radius': -1,  # # Radius in um to build channel neighborhood (-1 to include all channels in every neighborhood)
                                    'freq_min': 300,  # 'High-pass filter cutoff frequency'
                                    'freq_max': 6000, #'Low-pass filter cutoff frequency'
                                    'filter': True, 
                                    'whiten': True,  # Whether to do channel whitening as part of preprocessing
                                    'num_workers': 1, 
                                    'clip_size': 50, #'Number of samples per waveform',
                                    'detect_threshold': 3,
                                    'detect_interval': 10,  # Minimum number of timepoints between events detected on the same channel
                                    'tempdir': None #'Temporary directory for mountainsort (available for ms4 >= 1.0.2)s'
                                },
                'tridesclous': {
                                    'freq_min': 300.,   #'High-pass filter cutoff frequency'
                                    'freq_max': 6000.,#'Low-pass filter cutoff frequency'
                                    'detect_sign': -1,     #'Use -1 (negative) or 1 (positive) depending on the sign of the spikes in the recording',
                                    'detect_threshold': 5, #'Threshold for spike detection',
                                    'n_jobs' : 8,           #'Number of jobs (when saving ti binary) - default -1 (all cores)',
                                    'common_ref_removal': False,     #'remove common reference with median',
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
                'waveclus':  {
                                    'detect_threshold': 5,
                                    'detect_sign': -1,
                                    'feature_type': 'wav',
                                    'scales': 4,
                                    'min_clus': 20,
                                    'maxtemp': 0.251,
                                    'template_sdnum': 3,
                                    'enable_detect_filter': True,
                                    'enable_sort_filter': True,
                                    'detect_filter_fmin': 300,
                                    'detect_filter_fmax': 6000,
                                    'detect_filter_order': 4,
                                    'sort_filter_fmin': 300,
                                    'sort_filter_fmax': 6000,
                                    'sort_filter_order': 2,
                                    'mintemp': 0,
                                    'w_pre': 20,
                                    'w_post': 44,
                                    'alignment_window': 10,
                                    'stdmax': 50,
                                    'max_spk': 40000,
                                    'ref_ms': 1.5,
                                    'interpolation': True,
                                    'keep_good_only': True,
                                    'chunk_memory': '500M'},
                
                'ironclust':  {
                                    'detect_sign': -1,
                                    'adjacency_radius': 50,
                                    'adjacency_radius_out': 100,
                                    'detect_threshold': 3.5,
                                    'prm_template_name': '',
                                    'freq_min': 300,
                                    'freq_max': 6000,
                                    'merge_thresh': 0.985,
                                    'pc_per_chan': 9,
                                    'whiten': False,
                                    'filter_type': 'bandpass',
                                    'filter_detect_type': 'none',
                                    'common_ref_type': 'trimmean',
                                    'batch_sec_drift': 300,
                                    'step_sec_drift': 20,
                                    'knn': 30,
                                    'min_count': 30,
                                    'fGpu': True,
                                    'fft_thresh': 8,
                                    'fft_thresh_low': 0,
                                    'nSites_whiten': 16,
                                    'feature_type': 'gpca',
                                    'delta_cut': 1,
                                    'post_merge_mode': 1,
                                    'sort_mode': 1,
                                    'fParfor': False,
                                    'filter': True,
                                    'clip_pre': 0.25,
                                    'clip_post': 0.75,
                                    'merge_thresh_cc': 1,
                                    'nRepeat_merge': 3,
                                    'merge_overlap_thresh': 0.95,
                                    'n_jobs': 8,
                                    'chunk_duration': '1s',
                                    'progress_bar': True},
                
                'herdingspikes':   {
                                    'clustering_bandwidth': 5.5,
                                    'clustering_alpha': 5.5,
                                    'clustering_n_jobs': -1,
                                    'clustering_bin_seeding': True,
                                    'clustering_min_bin_freq': 16,
                                    'clustering_subset': None,
                                    'left_cutout_time': 0.3,
                                    'right_cutout_time': 1.8,
                                    'detect_threshold': 20,
                                    'probe_masked_channels': [],
                                    'probe_inner_radius': 70,
                                    'probe_neighbor_radius': 90,
                                    'probe_event_length': 0.26,
                                    'probe_peak_jitter': 0.2,
                                    't_inc': 100000,
                                    'num_com_centers': 1,
                                    'maa': 12,
                                    'ahpthr': 11,
                                    'out_file_name': 'HS2_detected',
                                    'decay_filtering': False,
                                    'save_all': False,
                                    'amp_evaluation_time': 0.4,
                                    'spk_evaluation_time': 1.0,
                                    'pca_ncomponents': 2,
                                    'pca_whiten': True,
                                    'freq_min': 300.0,
                                    'freq_max': 6000.0,
                                    'filter': True,
                                    'pre_scale': True,
                                    'pre_scale_value': 20.0,
                                    'filter_duplicates': True}
                
            }








"""
---------------------------Spike sorting---------------------------

"""
#Load the recordings
recording_loaded = si.load_extractor(rf"{working_dir}/{subject_name}\{recording_name}")

multirecording = recording_loaded.split_by('group')[0]
w = sw.plot_timeseries(multirecording,time_range=[10,15], segment_index=0)

print(f'Loaded channels ids: {recording_loaded.get_channel_ids()}')
print(f'Channel groups after loading: {recording_loaded.get_channel_groups()}')


print("Spike sorting starting")
# sorter_list = []
# sorter_name_list = []
# for sorter_name in param_sorter:
#     print(sorter_name)
#     sorter_list.append(ss.run_sorter(sorter_name,
#         recording=multirecording,
#         output_folder=f"{sorting_saving_dir}/{sorter_name}",
#         docker_image=True))
#     sorter_name_list.append(sorter_name)




sorter_list = []
sorter_name_list = []
for sorter_name, sorter_param in param_sorter.items():
    print(sorter_name)
    sorter_list.append(
        ss.run_sorter(
        sorter_name,
        recording=multirecording,
        output_folder=f"{sorting_saving_dir}/{sorter_name}",
        docker_image=True,verbose=True,
        **sorter_param))
    sorter_name_list.append(sorter_name)

print("Spike sorting done")





#Get rid of empty units
sorter_list_cleaned=[]


for sorter_result in sorter_list:
    sorter_list_cleaned.append(sorter_result.remove_empty_units())



print("getting rid of empty sorters")
# Liste des sorters vides à supprimer
sorters_empty = []

# Parcourir les sorters dans la liste avec l'indice
for index, sorter_result in enumerate(sorter_list_cleaned):
    # Vérifier si le sorter est vide
    if sorter_result.get_num_units() == 0:
        sorters_empty.append((index, sorter_result))  # Ajouter le tuple (indice, sorter vide) à la liste des sorters vides

# Supprimer les sorters vides de la liste
for i in sorters_empty:
    sorter_list_cleaned.remove(i[1])
    print(rf'{sorter_name_list[i[0]]} is empty and deleted')
    sorter_name_list.remove(sorter_name_list[i[0]])



#%% Section 2




"""
Export to PHY
"""
print("Exporting spike sorting results to Phy")

job_kwargs = dict(n_jobs=10, chunk_duration="1s", progress_bar=True)


for i in range(len(sorter_list_cleaned)):
    print('********************************')
    print(rf'********{sorter_name_list[i]}********')
    print('********************************')
    we = si.extract_waveforms(recording_loaded, sorter_list_cleaned[i], folder=rf'{working_dir}/{subject_name}/waveform_output/{recording_name}/{sorter_name_list[i]}', 
                              load_if_exists=False, overwrite=True,**job_kwargs)
    
    try :
        sexp.export_to_phy(we, output_folder=rf'{working_dir}/{subject_name}/export_phy/{recording_name}/{sorter_name_list[i]}', 
                       compute_amplitudes=False, compute_pc_features=False, copy_binary=True,remove_if_exists=True,
                       **job_kwargs)
    except : 
        print(rf'{sorter_name_list[i]} export skipped (no units?)')
        pass
    
print("Exporting to Phy done")





#%% Section 3



"""
Comparison

"""

print("Computing comparison")



mcmp = sc.compare_multiple_sorters(
    sorting_list=sorter_list_cleaned,
    name_list=sorter_name_list,
    verbose=True,
)

agr_2 = mcmp.get_agreement_sorting(minimum_agreement_count=2)



if agr_2.get_num_units() == 0:
    print('No unit in agreement')
else:
    print("Plotting comparison")
    w = sw.plot_multicomp_agreement(mcmp)
    w = sw.plot_multicomp_agreement_by_sorter(mcmp)
    
    print("Exporting to Phy comparison")
    
    we = si.extract_waveforms(recording_loaded, agr_2, folder=rf'{working_dir}/{subject_name}/waveform_output/{recording_name}/agreement', 
                              load_if_exists=False, overwrite=True,**job_kwargs)
    
    sexp.export_to_phy(we, output_folder=rf'{working_dir}/{subject_name}/export_phy/{recording_name}/agreement', 
                       compute_amplitudes=False, compute_pc_features=False, copy_binary=True,
                       **job_kwargs)
    
    print("Exporting to Phy comparison done")
    
    
    
    
    
    
