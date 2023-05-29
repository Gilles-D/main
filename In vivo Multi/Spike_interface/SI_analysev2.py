# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:47:01 2023

@author: MOCAP
"""

import spikeinterface as si

import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se 

import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.comparison as sc
import spikeinterface.exporters as sexp
import spikeinterface.widgets as sw


import probeinterface as pi
from probeinterface.plotting import plot_probe

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


import warnings
warnings.simplefilter("ignore")


"""
------------------PARAMETERS------------------

"""
probe_path=r'D:/ePhy/SI_Data/A1x16-Poly2-5mm-50s-177.json'
# probe_path=r'//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Buzsaki16.json'

# Data folder path
working_dir=r'D:\ePhy\Intan_Data'

# Saving Folder path
saving_dir=r"D:\ePhy\SI_Data"

subject_name="0012"

saving_name="0012_19_05_01"

freq_min = 300
freq_max = 6000

#Modifier en une boucle sur le dossier itan
# recordings = [  
# "D:/ePhy/Intan_Data/0012/0012_15_05_230515_172650/0012_15_05_230515_172650.rhd",
# "D:/ePhy/Intan_Data/0012/0012_15_05_230515_172650/0012_15_05_230515_172750.rhd",
# "D:/ePhy/Intan_Data/0012/0012_15_05_230515_172650/0012_15_05_230515_172850.rhd"]

recording = r"D:/ePhy/Intan_Data/0012/0012_19_05_230519_165017/0012_19_05_01.rhd"

sorting_saving_dir=rf'{working_dir}/{subject_name}/sorting_output/{saving_name}'

param_sorter = {
    # 'kilosort3': {
    #                                 'detect_threshold': 6,
    #                                 'projection_threshold': [9, 9],
                                #     'preclust_threshold': 8,
                                #     'car': True,
                                #     'minFR': 0.2,
                                #     'minfr_goodchannels': 0.2,
                                #     'nblocks': 5,
                                #     'sig': 20,
                                #     'freq_min': 300,
                                #     'sigmaMask': 30,
                                #     'nPCs': 3,
                                #     'ntbuff': 64,
                                #     'nfilt_factor': 4,
                                #     'do_correction': True,
                                #     'NT': None,
                                #     'wave_length': 61,
                                #     'keep_good_only': False,
                                # },
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
                'tridesclous': {
                                    'freq_min': 300.,
                                    'freq_max': 6000.,
                                    'detect_sign': -1,
                                    'detect_threshold': 3,
                                    'common_ref_removal': True,
                                    'nested_params': None,
                                },
                'spykingcircus': {
                                    'detect_sign': -1,  # -1 - 1 - 0
                                    'adjacency_radius': 100,  # Channel neighborhood adjacency radius corresponding to geom file
                                    'detect_threshold': 3,  # Threshold for detection
                                    'template_width_ms': 3,  # Spyking circus parameter
                                    'filter': True,
                                    'merge_spikes': True,
                                    'auto_merge': 0.75,
                                    'num_workers': None,
                                    'whitening_max_elts': 1000,  # I believe it relates to subsampling and affects compute time
                                    'clustering_max_elts': 10000,  # I believe it relates to subsampling and affects compute time
                                }
            }














# #Concatenate all the recording files
# recordings_list=[]
# for recording_file in recordings:
#     recording = se.read_intan(recording_file,stream_id='0')
#     recording.annotate(is_filtered=False)
#     recordings_list.append(recording)


# multirecording = si.concatenate_recordings(recordings_list)

multirecording=se.read_intan(recording,stream_id='0')


#Set the probe
probe = pi.io.read_probeinterface(probe_path)
probe = probe.probes[0]
multirecording = multirecording.set_probe(probe)
# plot_probe(probe, with_channel_index=True, with_device_index=True)


"""------------------Defective sites exclusion------------------"""
#Check the defective sites
# sw.plot_timeseries(multirecording, channel_ids=multirecording.get_channel_ids(),time_range=[0,15])



# excluded_sites = ['2','3','4','9','10','11','12','14','15']
excluded_sites = []

#Exclude defective sites
# multirecording.set_channel_groups(1, [])

# multirecording.set_channel_groups(1, ['4','9','10','11','12','14','15'])
multirecording.set_channel_groups(1, excluded_sites)
multirecording = multirecording.split_by('group')[0]

# w = sw.plot_timeseries(multirecording,time_range=[0,15])


"""------------------Pre Processing------------------"""
#Bandpass filter
recording_f = spre.bandpass_filter(multirecording, freq_min=freq_min, freq_max=freq_max)
# w = sw.plot_timeseries(recording_f,time_range=[10,15], segment_index=0)


#Median common ref
recording_cmr = spre.common_reference(recording_f, reference='global', operator='median')
# w = sw.plot_timeseries(recording_cmr,time_range=[10,15], segment_index=0)



"""
---------------------------Spike sorting---------------------------

"""
#Load the recordings
recording_loaded = recording_cmr

multirecording = recording_loaded.split_by('group')[0]
w = sw.plot_timeseries(multirecording,time_range=[10,15], segment_index=0)

print(f'Loaded channels ids: {recording_loaded.get_channel_ids()}')
print(f'Channel groups after loading: {recording_loaded.get_channel_groups()}')



sorter_list = []
sorter_name_list = []

for sorter_name, sorter_param in param_sorter.items():
    print(sorter_name)
    sorter_list.append(ss.run_sorter(sorter_name,
        recording=multirecording,
        output_folder=f"{sorting_saving_dir}/{sorter_name}",
        docker_image=True,
        **sorter_param))
    sorter_name_list.append(sorter_name)
