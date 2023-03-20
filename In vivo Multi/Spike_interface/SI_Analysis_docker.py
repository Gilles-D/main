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


"""
---------------------------PARAMETERS---------------------------

"""

use_docker = True

# Working folder path
working_dir=r'C:\Users\MOCAP\Desktop\datasi'

subject_name="Test_Gustave"
recording_name='Gustave_09_03_baseline2'


sorting_saving_dir=rf'{working_dir}/{subject_name}/sorting_output/{recording_name}'

param_sorter = {'kilosort3': {
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
                                    'freq_min': 400.,
                                    'freq_max': 5000.,
                                    'detect_sign': -1,
                                    'detect_threshold': 5,
                                    'common_ref_removal': False,
                                    'nested_params': None,
                                },
                'spykingcircus': {
                                    'detect_sign': -1,  # -1 - 1 - 0
                                    'adjacency_radius': 100,  # Channel neighborhood adjacency radius corresponding to geom file
                                    'detect_threshold': 6,  # Threshold for detection
                                    'template_width_ms': 3,  # Spyking circus parameter
                                    'filter': True,
                                    'merge_spikes': True,
                                    'auto_merge': 0.75,
                                    'num_workers': None,
                                    'whitening_max_elts': 1000,  # I believe it relates to subsampling and affects compute time
                                    'clustering_max_elts': 10000,  # I believe it relates to subsampling and affects compute time
                                }
            }

"""
---------------------------Spike sorting---------------------------

"""
#Load the recordings
recording_loaded = si.load_extractor(rf"{working_dir}/{subject_name}\raw\raw si\{recording_name}")

multirecording = recording_loaded.split_by('group')[0]
w = sw.plot_timeseries(multirecording,time_range=[10,15], segment_index=0)

print(f'Loaded channels ids: {recording_loaded.get_channel_ids()}')
print(f'Channel groups after loading: {recording_loaded.get_channel_groups()}')


#Sorting
#ss.installed_sorters()

# sorting_outputs = ss.run_sorters(sorter_list=["kilosort3"],
#                                  recording_dict_or_list={"group0": recording_loaded, "group1": recording_loaded},
#                                  working_folder=sorting_saving_dir,
#                                  verbose=True,
#                                  engine="joblib",
#                                  engine_kwargs={'n_jobs': 1},
#                                  docker_images=True)

sorter_list = []
sorter_name_list = []
for sorter_name, sorter_param in param_sorter.items():
    sorter_list.append(ss.run_sorter(sorter_name,
        recording=multirecording,
        output_folder=f"{sorting_saving_dir}/{sorter_name}",
        docker_image=True,
        **sorter_param))
    sorter_name_list.append(sorter_name)

# cmp_HS_TDC = sc.compare_two_sorters(
#     sorting1=sorting,
#     sorting2=sorting_1,
#     sorting1_name='kilo',
#     sorting2_name='mountain',
# )
mcmp = sc.compare_multiple_sorters(
    sorting_list=sorter_list,
    name_list=sorter_name_list,
    verbose=True,
)
agr_2 = mcmp.get_agreement_sorting(minimum_agreement_count=2)

# sw.plot_agreement_matrix(cmp_HS_TDC)

#Remove empty units
TDC_output = sorting_outputs[('group0','tridesclous')]
TDC_output = TDC_output.remove_empty_units()
print(f'Sorter found {len(TDC_output.get_unit_ids())} non-empty units')

#Save sorting output
TDC_output_saved = TDC_output.save(folder=rf'{sorting_saving_dir}\TDC_output/') 


#Raster plot
w_rs = sw.plot_rasters(TDC_output)


"""
---------------------------Waveform extraction---------------------------

"""

job_kwargs = dict(n_jobs=10, chunk_duration="1s", progress_bar=True)

#Waveform extraction only 500 for each clsuter
we = si.extract_waveforms(recording_loaded, TDC_output, folder=rf'{working_dir}/{subject_name}/waveform_output/{recording_name}', 
                          load_if_exists=False, overwrite=True,**job_kwargs)
print(we)

waveforms0 = we.get_waveforms(unit_id=0)
print(f"Waveforms shape: {waveforms0.shape}")
template0 = we.get_template(unit_id=0)
print(f"Template shape: {template0.shape}")
all_templates = we.get_all_templates()
print(f"All templates shape: {all_templates.shape}")

w = sw.plot_unit_templates(we)

for unit in TDC_output.get_unit_ids():
    waveforms = we.get_waveforms(unit_id=unit)
    spiketrain = TDC_output.get_unit_spike_train(unit)
    print(f"Unit {unit} - num waveforms: {waveforms.shape[0]} - num spikes: {len(spiketrain)}")
    
    
#Waveform extraction all spikes for each clsuter  
we_all = si.extract_waveforms(recording_loaded, TDC_output, folder=rf'{working_dir}/{subject_name}/waveform_output_all/{recording_name}', 
                              max_spikes_per_unit=None,
                              overwrite=True,
                              **job_kwargs)

for unit in TDC_output.get_unit_ids():
    waveforms = we_all.get_waveforms(unit_id=unit)
    spiketrain = TDC_output.get_unit_spike_train(unit)
    print(f"Unit {unit} - num waveforms: {waveforms.shape[0]} - num spikes: {len(spiketrain)}")
    


"""
---------------------------Post processing---------------------------

"""
sorting = TDC_output


"""
Sparsity
"""

# example: radius
sparsity_radius = spost.get_template_channel_sparsity(we, method="radius", radius_um=50)

sparsity_radius = si.core.ChannelSparsity.from_radius(we, radius_um=50, peak_sign='neg')


# example: best
sparsity_best = spost.get_template_channel_sparsity(we, method="best_channels", num_channels=4)

sparsity_best = si.core.ChannelSparsity.from_best_channels(we, num_channels=4, peak_sign='neg')


sw.plot_unit_templates(we, sparsity=sparsity_radius)
sw.plot_unit_templates(we, sparsity=sparsity_best)

"""
PCA scores
"""
# spost.compute_principal_components?
pc = spost.compute_principal_components(we, n_components=3,
                                        sparsity=sparsity_best, 
                                        load_if_exists=False,
                                        n_jobs=job_kwargs["n_jobs"], 
                                        progress_bar=job_kwargs["progress_bar"])

pc0 = pc.get_projections(unit_id=0)
print(f"PC scores shape: {pc0.shape}")
all_labels, all_pcs = pc.get_all_projections()
print(f"All PC scores shape: {all_pcs.shape}")



"""
WaveformExtensions
"""
we.get_available_extension_names()
pc = we.load_extension("principal_components")
print(pc)

all_labels, all_pcs = pc.get_data()
print(all_pcs.shape)


"""
SpikeAmplitudes
"""


amplitudes = spost.compute_spike_amplitudes(we, outputs="by_unit", load_if_exists=False, 
                                            **job_kwargs)
amplitudes[0]

sw.plot_amplitudes(we)


"""
Compute unit and spike locations
"""
unit_locations = spost.compute_unit_locations(we, method="monopolar_triangulation", load_if_exists=True)
spike_locations = spost.compute_spike_locations(we, method="center_of_mass", load_if_exists=True,
                                                **job_kwargs)

sw.plot_unit_locations(we)
sw.plot_spike_locations(we, max_spikes_per_unit=300)


"""
Compute correlograms
"""
ccgs, bins = spost.compute_correlograms(we)
sw.plot_autocorrelograms(we, unit_ids=sorting.unit_ids[:])
sw.plot_crosscorrelograms(we, unit_ids=sorting.unit_ids[:])

"""
Compute template similarity
"""
similarity = spost.compute_template_similarity(we)


"""
Compute template metrics
"""
print(spost.get_template_metric_names())
template_metrics = spost.calculate_template_metrics(we)
display(template_metrics)

sw.plot_template_metrics(we, include_metrics=["peak_to_valley", "half_width"])


"""
---------------------------Quality metrics and curation---------------------------

"""

print(sqm.get_quality_metric_list())
print(sqm.get_quality_pca_metric_list())

qm = sqm.compute_quality_metrics(we, sparsity=sparsity_radius, verbose=True, 
                                 n_jobs=job_kwargs["n_jobs"])

display(qm)

sw.plot_quality_metrics(we, include_metrics=["amplitude_cutoff", "presence_ratio", "isi_violations_ratio", "snr"])

"""
Automatic curation based on quality metrics
"""
isi_violations_rate = 0.2
amp_cutoff_thresh = 0.1

our_query = f"amplitude_cutoff < {amp_cutoff_thresh} & isi_violations_rate < {isi_violations_rate}"
print(our_query)

keep_units = qm.query(our_query)
keep_unit_ids = keep_units.index.values
