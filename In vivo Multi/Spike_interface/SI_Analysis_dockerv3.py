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

#%% RHD file loading

probe_path=r'D:/ePhy/SI_Data/A1x16-Poly2-5mm-50s-177.json'   #INTAN Optrode
# probe_path = 'D:/ePhy/SI_Data/Buzsaki16.json' #INTAN Buzsaki16


freq_min = 300
freq_max = 6000

# Data folder path
working_dir=r'D:\ePhy\Intan_Data'

# Saving Folder path
saving_dir=r"D:\ePhy\SI_Data"

subject_name="0012"

saving_name="0012_29_06"

#Modifier en une boucle sur le dossier itan
#Modifier en une boucle sur le dossier itan
# recordings = [  
# "D:/ePhy/Intan_Data/0012/05_24/0014_24_05_230524_163229/0014_24_05_230524_163229.rhd",
# "D:/ePhy/Intan_Data/0012/05_24/0014_24_05_230524_163605/0014_24_05_230524_163605.rhd",
# "D:/ePhy/Intan_Data/0012/05_24/0014_24_05_230524_163928/0014_24_05_230524_163928.rhd",
# "D:/ePhy/Intan_Data/0012/05_24/0014_24_05_230524_164414/0014_24_05_230524_164414.rhd"]


recordings=['D:/ePhy/Intan_Data/0012/06_28/0012_28_06_230628_170325/0012_28_06_230628_170325.rhd',#2
'D:/ePhy/Intan_Data/0012/06_28/0012_28_06_230628_171510/0012_28_06_230628_171510.rhd',#4
'D:/ePhy/Intan_Data/0012/06_28/0012_28_06_230628_172006/0012_28_06_230628_172006.rhd',#5
'D:/ePhy/Intan_Data/0012/06_28/0012_28_06_230628_173325/0012_28_06_230628_173325.rhd',#6
'D:/ePhy/Intan_Data/0012/06_28/0012_28_06_230628_174731/0012_28_06_230628_174731.rhd']#8


# ['D:/ePhy/Intan_Data/0012/06_08/0012_08_06_230608_154253/0012_08_06_230608_154253.rhd',
#             'D:/ePhy/Intan_Data/0012/06_08/0012_08_06_230608_155001/0012_08_06_230608_155001.rhd',
#             'D:/ePhy/Intan_Data/0012/06_08/0012_08_06_230608_161048/0012_08_06_230608_161048.rhd',
#             'D:/ePhy/Intan_Data/0012/06_08/0012_08_06_230608_163818/0012_08_06_230608_163818.rhd',
#             'D:/ePhy/Intan_Data/0012/06_08/0012_08_06_230608_164712/0012_08_06_230608_164712.rhd'
    
    
    
    
    
#     ]



"""
["D:/ePhy/Intan_Data/0012/06_08/0012_08_06_230608_153639/0012_08_06_230608_154139.rhd",
"D:/ePhy/Intan_Data/0012/06_08/0012_08_06_230608_153639/0012_08_06_230608_153639.rhd",
"D:/ePhy/Intan_Data/0012/06_08/0012_08_06_230608_153639/0012_08_06_230608_153739.rhd",
"D:/ePhy/Intan_Data/0012/06_08/0012_08_06_230608_153639/0012_08_06_230608_153839.rhd",
"D:/ePhy/Intan_Data/0012/06_08/0012_08_06_230608_153639/0012_08_06_230608_153939.rhd",
"D:/ePhy/Intan_Data/0012/06_08/0012_08_06_230608_153639/0012_08_06_230608_154039.rhd",

'D:/ePhy/Intan_Data/0012/06_08/0012_08_06_230608_154253/0012_08_06_230608_154253.rhd',

'D:/ePhy/Intan_Data/0012/06_08/0012_08_06_230608_155001/0012_08_06_230608_155001.rhd',

'D:/ePhy/Intan_Data/0012/06_08/0012_08_06_230608_155941/0012_08_06_230608_155941.rhd',

'D:/ePhy/Intan_Data/0012/06_08/0012_08_06_230608_161048/0012_08_06_230608_161048.rhd',

"D:/ePhy/Intan_Data/0012/06_08/0012_08_06_230608_161806/0012_08_06_230608_162808.rhd",
"D:/ePhy/Intan_Data/0012/06_08/0012_08_06_230608_161806/0012_08_06_230608_161806.rhd",

'D:/ePhy/Intan_Data/0012/06_08/0012_08_06_230608_163818/0012_08_06_230608_163818.rhd',

'D:/ePhy/Intan_Data/0012/06_08/0012_08_06_230608_164712/0012_08_06_230608_164712.rhd',

'D:/ePhy/Intan_Data/0012/06_08/0012_08_06_230608_165232/0012_08_06_230608_165232.rhd',

'D:/ePhy/Intan_Data/0012/06_08/0012_08_06_230608_165941/0012_08_06_230608_165941.rhd']
"""



#Concatenate all the recording files
recordings_list=[]
for recording_file in recordings:
    recording = se.read_intan(recording_file,stream_id='0')
    recording.annotate(is_filtered=False)
    recordings_list.append(recording)

multirecording = si.concatenate_recordings(recordings_list)

#Set the probe
probe = pi.io.read_probeinterface(probe_path)
probe = probe.probes[0]
multirecording = multirecording.set_probe(probe)
# plot_probe(probe, with_channel_index=True, with_device_index=True)


"""------------------Defective sites exclusion------------------"""
#Check the defective sites
# sw.plot_timeseries(multirecording, channel_ids=multirecording.get_channel_ids(),time_range=[0,15])



#excluded_sites = ['2','4','9','10','11','12','14','15']
# excluded_sites = ['6','7','9','10','11','14','15']
excluded_sites = ['6','7','9','10','11']
#excluded_sites = []

#Exclude defective sites
# multirecording.set_channel_groups(1, [])

# multirecording.set_channel_groups(1, ['4','9','10','11','12','14','15'])
multirecording.set_channel_groups(1, excluded_sites)
multirecording = multirecording.split_by('group')[0]

# w = sw.plot_timeseries(multirecording,time_range=[10,15])


"""------------------Pre Processing------------------"""
#Bandpass filter
recording_f = spre.bandpass_filter(multirecording, freq_min=freq_min, freq_max=freq_max)
# w = sw.plot_timeseries(recording_f,time_range=[10,15], segment_index=0)


#Median common ref
recording_cmr = spre.common_reference(recording_f, reference='global', operator='median')
# w = sw.plot_timeseries(recording_cmr,time_range=[10,15], segment_index=0)


rec_binary = recording_cmr.save(folder=rf'C:\Users\MOCAP\Desktop\temp\{saving_name}_test', n_jobs=1, progress_bar=True, chunk_duration='1s')

#%% Section 1

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

#%% Section 2

# list_sorter_output = ["C:/Users/MOCAP/Desktop/temp/_7_22_05_01/ironclust",
# "C:/Users/MOCAP/Desktop/temp/_7_22_05_01/mountainsort4",
# "C:/Users/MOCAP/Desktop/temp/_7_22_05_01/spykingcircus",
# "C:/Users/MOCAP/Desktop/temp/_7_22_05_01/tridesclous",
# "C:/Users/MOCAP/Desktop/temp/_7_22_05_01/waveclus"]
# sorter_list_cleaned=[]

# for sorter in list_sorter_output:
#     sorter_list_cleaned.append(ss.read_sorter_folder(sorter))



# ss.read_sorter_folder(r'C:\Users\MOCAP\Desktop\temp\_7_22_05_01\mountainsort4')

"""
Export to PHY
"""
print("Exporting spike sorting results to Phy")

job_kwargs = dict(n_jobs=1, chunk_duration="1s", progress_bar=True)


for i in range(len(sorter_list_cleaned)):
    print('********************************')
    print(rf'********{sorter_name_list[i]}********')
    print('********************************')
    we = si.extract_waveforms(recording_cmr, sorter_list_cleaned[i], folder=rf'C:\Users\MOCAP\Desktop\temp\{saving_name}\wf_export\{sorter_name[i]}', 
                              load_if_exists=False, overwrite=True,**job_kwargs)
    
    try :
        sexp.export_to_phy(we, output_folder=rf'C:\Users\MOCAP\Desktop\temp\{saving_name}\phy_export\{sorter_name_list[i]}', 
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
    
    we = si.extract_waveforms(recording_cmr, agr_2, folder=rf'{working_dir}/{subject_name}/waveform_output/{saving_name}/agreement', 
                              load_if_exists=False, overwrite=True,**job_kwargs)
    
    sexp.export_to_phy(we, output_folder=rf'{working_dir}/{subject_name}/export_phy/{saving_name}/agreement', 
                       compute_amplitudes=False, compute_pc_features=False, copy_binary=True,
                       **job_kwargs)
    
    print("Exporting to Phy comparison done")
    
    
    
    
    
    
