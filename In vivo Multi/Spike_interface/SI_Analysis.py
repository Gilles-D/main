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

import spikeinterface_gui as sigui

import warnings
warnings.simplefilter("ignore")


"""
---------------------------PARAMETERS---------------------------

"""


# Working folder path
working_dir=r'\\equipe2-nas1\Gilles.DELBECQ\Data\ePhy\Février2023'

subject_name="Test_Gustave"
recording_name='Gustave_09_03_baseline2'


sorting_saving_dir=rf'{working_dir}/{subject_name}/sorting_output/{recording_name}'



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

sorting_outputs = ss.run_sorters(sorter_list=["tridesclous"],
                                 recording_dict_or_list={"group0": recording_loaded, "group1": recording_loaded},
                                 working_folder=sorting_saving_dir,
                                 verbose=True,
                                 engine="joblib",
                                 engine_kwargs={'n_jobs': 1})

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
