# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 10:50:23 2022

@author: Gilles.DELBECQ
"""

import spikeinterface as si
import spikeinterface.extractors as se 
import spikeinterface.toolkit as st
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw

import os

import matplotlib.pyplot as plt
import numpy as np

import probeinterface as pi

# %matplotlib notebook

"""Setup"""
# file path
recording_file = r'//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/13-09/raw/06_02_0001_20000Hz.rbf'

recordings = [
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/12-09/raw/06_01_0001_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/12-09/raw/06_01_0002_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/12-09/raw/06_01_0003_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/12-09/raw/06_01_0004_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/12-09/raw/06_01_0005_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/12-09/raw/06_01_0006_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/12-09/raw/06_01_0007_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/12-09/raw/06_01_0008_20000Hz.rbf"
    
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/13-09/raw/06_02_0016_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/13-09/raw/06_02_0001_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/13-09/raw/06_02_0002_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/13-09/raw/06_02_0003_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/13-09/raw/06_02_0004_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/13-09/raw/06_02_0005_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/13-09/raw/06_02_0006_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/13-09/raw/06_02_0007_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/13-09/raw/06_02_0008_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/13-09/raw/06_02_0009_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/13-09/raw/06_02_0010_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/13-09/raw/06_02_0011_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/13-09/raw/06_02_0012_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/13-09/raw/06_02_0013_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/13-09/raw/06_02_0014_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/13-09/raw/06_02_0015_20000Hz.rbf",


# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/14-09/raw/06_03_0011_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/14-09/raw/06_03_0001_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/14-09/raw/06_03_0002_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/14-09/raw/06_03_0003_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/14-09/raw/06_03_0004_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/14-09/raw/06_03_0005_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/14-09/raw/06_03_0006_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/14-09/raw/06_03_0007_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/14-09/raw/06_03_0008_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/14-09/raw/06_03_0009_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/14-09/raw/06_03_0010_20000Hz.rbf",

# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/12-09/raw/08_01_0001_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/12-09/raw/08_01_0002_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/12-09/raw/08_01_0003_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/12-09/raw/08_01_0004_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/12-09/raw/08_01_0005_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/12-09/raw/08_01_0006_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/12-09/raw/08_01_0007_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/12-09/raw/08_01_0008_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/12-09/raw/08_01_0009_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/12-09/raw/08_01_0010_20000Hz.rbf"

# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/13-09/raw/08_02_0001_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/13-09/raw/08_02_0002_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/13-09/raw/08_02_0003_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/13-09/raw/08_02_0004_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/13-09/raw/08_02_0005_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/13-09/raw/08_02_0006_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/13-09/raw/08_02_0007_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/13-09/raw/08_02_0008_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/13-09/raw/08_02_0009_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/13-09/raw/08_02_0010_20000Hz.rbf"

# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/14-09/raw/08_03_0001_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/14-09/raw/08_03_0002_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/14-09/raw/08_03_0003_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/14-09/raw/08_03_0004_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/14-09/raw/08_03_0005_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/14-09/raw/08_03_0006_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/14-09/raw/08_03_0007_20000Hz.rbf",
# "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/14-09/raw/08_03_0008_20000Hz.rbf"
]


Experience_identifier='0008_02'
Date=recording_file.split('/')[-3]

# Working folder path
working_dir=fr'D:\Working_Dir\Ephy\Cohorte_1\{Date}\{Experience_identifier}'

if os.path.exists(working_dir)==False:
    os.makedirs(working_dir)
os.chdir(working_dir)


# parameters to load the bin/dat format
num_channels = 16
sampling_frequency = 20000
gain_to_uV = 0
offset_to_uV = 0
dtype = "float64"
time_axis = 0


"""File loading and setup"""
#Load multiple files
recording = se.BinaryRecordingExtractor(recordings, sampling_frequency, num_chan=num_channels, dtype=dtype)


#load the file and create object recording

# recording = si.read_binary(recording_file, num_chan=num_channels, sampling_frequency=sampling_frequency,
#                            dtype=dtype, gain_to_uV=gain_to_uV, offset_to_uV=offset_to_uV, 
#                            time_axis=time_axis)

print(recording)

recording.annotate(is_filtered=False)

channel_ids = recording.get_channel_ids()
fs = recording.get_sampling_frequency()
num_chan = recording.get_num_channels()
num_segments = recording.get_num_segments()

print(f'Channel ids: {channel_ids}')
print(f'Sampling frequency: {fs}')
print(f'Number of channels: {num_chan}')
print(f"Number of segments: {num_segments}")

# explore the traces here
# recording.get_traces()

#Set the probe
probe_path=r'D:/Working_Dir/Ephy/CM16_Buz_Sparse.json'
probe = pi.io.read_probeinterface(probe_path)
probe = probe.probes[0]
recording = recording.set_probe(probe)



"""Defective sites exclusion"""
#Check the defective sites
sw.plot_timeseries(recording, channel_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],time_range=[0,150],segment_index=5)

#3 7 11 15 are defective

#Exclude defective sites
# recording.set_channel_groups(1, [0,2,3,6,7,8,9,11,12,13,14,15])
# recording = recording.split_by('group')[1]
# w = sw.plot_timeseries(recording,time_range=[0,30])





"""Pre Processing"""
#Bandpass filter
recording_f = st.bandpass_filter(recording, freq_min=300, freq_max=4000)
w = sw.plot_timeseries(recording_f,time_range=[0,30],segment_index=5)


#Median common ref
recording_cmr = st.common_reference(recording_f, reference='global', operator='median')
w = sw.plot_timeseries(recording_cmr,time_range=[0,30],segment_index=5)



"""Save processed recordings"""
#Saving raw file
recording.save(folder="raw", progress_bar=True, n_jobs=1, total_memory="100M")

#Savinf processed file
recording_processed = recording_cmr 
recording_processed.set_annotation(annotation_key='bandpass_filter', value=(300,4000))
recording_processed.set_annotation(annotation_key='common_reference', value='median')
recording.annotate(is_filtered=True)
recording_processed.save(folder="preprocessed", progress_bar=True, n_jobs=1, total_memory="100M")




sorting_TDC = ss.run_sorter('tridesclous', recording, output_folder='results_TDC', 
                        remove_existing_folder=True, detect_threshold=5, verbose=True)
print(f'Tri des clous found {len(sorting_TDC.get_unit_ids())} units')

we = si.extract_waveforms(recording, sorting_TDC, folder="waveforms_si", progress_bar=True,
                          n_jobs=1, total_memory="500M", overwrite=True)
print(we)

unit_id0 = sorting_TDC.unit_ids[0]
wavefroms = we.get_waveforms(unit_id0)
print(wavefroms.shape)

template = we.get_template(unit_id0)
print(template.shape)