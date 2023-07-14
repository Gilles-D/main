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

import pandas as pd



def concatenate_preprocessing(recordings,saving_dir,saving_name,probe_path,excluded_sites,freq_min=300,freq_max=6000,Plotting=True):
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
    

    return rec_binary



"""
---------------------------PARAMETERS---------------------------

"""

use_docker = True

param_sorter = {
    # Parameters from all sorters can be retrieved with these functions:
    # params = ss.get_default_sorter_params('spykingcircus')
    # print("Parameters:\n", params)
    # desc = ss.get_sorter_params_description('spykingcircus')
    # print("Descriptions:\n", desc)
             
    
    # 'kilosort3': {
    #                                 'detect_threshold': 6,
    #                                 'projection_threshold': [9, 9],
    #                                 'preclust_threshold': 8,
    #                                 'car': True,
    #                                 'minFR': 0.2,
    #                                 'minfr_goodchannels': 0.2,
    #                                 'nblocks': 5,
    #                                 'sig': 20,
    #                                 'freq_min': 300,
    #                                 'sigmaMask': 30,
    #                                 'nPCs': 3,
    #                                 'ntbuff': 64,
    #                                 'nfilt_factor': 4,
    #                                 'do_correction': True,
    #                                 'NT': None,
    #                                 'wave_length': 61,
    #                                 'keep_good_only': False,
    #                             },
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
                                },
                # 'waveclus':  {
                #                     'detect_threshold': 5,
                #                     'detect_sign': -1,
                #                     'feature_type': 'wav',
                #                     'scales': 4,
                #                     'min_clus': 20,
                #                     'maxtemp': 0.251,
                #                     'template_sdnum': 3,
                #                     'enable_detect_filter': True,
                #                     'enable_sort_filter': True,
                #                     'detect_filter_fmin': 300,
                #                     'detect_filter_fmax': 6000,
                #                     'detect_filter_order': 4,
                #                     'sort_filter_fmin': 300,
                #                     'sort_filter_fmax': 6000,
                #                     'sort_filter_order': 2,
                #                     'mintemp': 0,
                #                     'w_pre': 20,
                #                     'w_post': 44,
                #                     'alignment_window': 10,
                #                     'stdmax': 50,
                #                     'max_spk': 40000,
                #                     'ref_ms': 1.5,
                #                     'interpolation': True,
                #                     'keep_good_only': True,
                #                     'chunk_memory': '500M'},
                
                # 'ironclust':  {
                #                     'detect_sign': -1,
                #                     'adjacency_radius': 50,
                #                     'adjacency_radius_out': 100,
                #                     'detect_threshold': 3.5,
                #                     'prm_template_name': '',
                #                     'freq_min': 300,
                #                     'freq_max': 6000,
                #                     'merge_thresh': 0.985,
                #                     'pc_per_chan': 9,
                #                     'whiten': False,
                #                     'filter_type': 'bandpass',
                #                     'filter_detect_type': 'none',
                #                     'common_ref_type': 'trimmean',
                #                     'batch_sec_drift': 300,
                #                     'step_sec_drift': 20,
                #                     'knn': 30,
                #                     'min_count': 30,
                #                     'fGpu': True,
                #                     'fft_thresh': 8,
                #                     'fft_thresh_low': 0,
                #                     'nSites_whiten': 16,
                #                     'feature_type': 'gpca',
                #                     'delta_cut': 1,
                #                     'post_merge_mode': 1,
                #                     'sort_mode': 1,
                #                     'fParfor': False,
                #                     'filter': True,
                #                     'clip_pre': 0.25,
                #                     'clip_post': 0.75,
                #                     'merge_thresh_cc': 1,
                #                     'nRepeat_merge': 3,
                #                     'merge_overlap_thresh': 0.95,
                #                     'n_jobs': 8,
                #                     'chunk_duration': '1s',
                #                     'progress_bar': True},
                
                # 'herdingspikes':   {
                #                     'clustering_bandwidth': 5.5,
                #                     'clustering_alpha': 5.5,
                #                     'clustering_n_jobs': -1,
                #                     'clustering_bin_seeding': True,
                #                     'clustering_min_bin_freq': 16,
                #                     'clustering_subset': None,
                #                     'left_cutout_time': 0.3,
                #                     'right_cutout_time': 1.8,
                #                     'detect_threshold': 20,
                #                     'probe_masked_channels': [],
                #                     'probe_inner_radius': 70,
                #                     'probe_neighbor_radius': 90,
                #                     'probe_event_length': 0.26,
                #                     'probe_peak_jitter': 0.2,
                #                     't_inc': 100000,
                #                     'num_com_centers': 1,
                #                     'maa': 12,
                #                     'ahpthr': 11,
                #                     'out_file_name': 'HS2_detected',
                #                     'decay_filtering': False,
                #                     'save_all': False,
                #                     'amp_evaluation_time': 0.4,
                #                     'spk_evaluation_time': 1.0,
                #                     'pca_ncomponents': 2,
                #                     'pca_whiten': True,
                #                     'freq_min': 300.0,
                #                     'freq_max': 6000.0,
                #                     'filter': True,
                #                     'pre_scale': True,
                #                     'pre_scale_value': 20.0,
                #                     'filter_duplicates': True}
                
             }



"""
---------------------------Spike sorting---------------------------

"""
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
        recording=rec_binary,
        output_folder=rf'C:\Users\MOCAP\Desktop\temp\{saving_name}\sorter_output\{sorter_name}',
        docker_image=True,verbose=False,
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

sorter_list_cleaned







probe_path=r'D:/ePhy/SI_Data/A1x16-Poly2-5mm-50s-177.json'   #INTAN Optrode
# probe_path = 'D:/ePhy/SI_Data/Buzsaki16.json'              #INTAN Buzsaki16

# Saving Folder path
saving_dir=r"D:/ePhy/SI_Data/concatenated_signals"
saving_name="0012_12_07"

excluded_sites = ['6','7','9','10','11']


recordings=["D:/ePhy/Intan_Data/0012/07_03/0012_07_03_File_06.rhd",
"D:/ePhy/Intan_Data/0012/07_03/0012_07_03_File_01.rhd",
"D:/ePhy/Intan_Data/0012/07_03/0012_07_03_File_02.rhd",
"D:/ePhy/Intan_Data/0012/07_03/0012_07_03_File_03.rhd",
"D:/ePhy/Intan_Data/0012/07_03/0012_07_03_File_04.rhd",
"D:/ePhy/Intan_Data/0012/07_03/0012_07_03_File_05.rhd"
    ]



concatenate_preprocessing(recordings,saving_dir,saving_name,probe_path,excluded_sites,Plotting=True)