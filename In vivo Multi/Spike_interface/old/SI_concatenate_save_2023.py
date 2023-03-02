# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 14:35:52 2023

@author: Gilles.DELBECQ
"""

import spikeinterface as si
import spikeinterface.extractors as se 
# import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
# import spikeinterface.postprocessing as spost
# import spikeinterface.qualitymetrics as sqm
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


probe_path=r'\\equipe2-nas1/Gilles.DELBECQ/Data/ePhy/A1x16-Poly2-5mm-50s-177.json'

# Working folder path
working_dir=r'\\equipe2-nas1\Gilles.DELBECQ\Data\ePhy\Février2023\testephy/'

if os.path.exists(working_dir)==False:
    os.makedirs(working_dir)
os.chdir(working_dir)

freq_min = 300
freq_max = 3000

recordings = [  
"//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Février2023/testephy/004_230221_152259/004_230221_152259.rhd",
"//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Février2023/testephy/004_230221_152259/004_230221_152359.rhd",
"//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Février2023/testephy/004_230221_152259/004_230221_152459.rhd",
"//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Février2023/testephy/004_230221_152259/004_230221_152559.rhd",
"//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Février2023/testephy/004_230221_152259/004_230221_152659.rhd",
"//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Février2023/testephy/004_230221_152259/004_230221_152759.rhd"
]

# recordings=[
# '//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Février2023/testephy/test ephy_230221_145224/test ephy_230221_145224.rhd'
# ]


recordings_list=[]
for recording_file in recordings:
    recording = si.extractors.read_intan(recording_file)
    recording.annotate(is_filtered=False)
    recordings_list.append(recording)

multirecording = si.concatenate_recordings(recordings_list)



#Set the probe
probe = pi.io.read_probeinterface(probe_path)
probe = probe.probes[0]
multirecording = multirecording.set_probe(probe)
plot_probe(probe, with_channel_index=True, with_device_index=True)

"""Defective sites exclusion"""
#Check the defective sites
sw.plot_timeseries(multirecording, channel_ids=multirecording.get_channel_ids(),time_range=[0,15])


#Exclude defective sites
#multirecording.set_channel_groups(1, [8,10,11,13,14,15])
multirecording.set_channel_groups(1, ['4','9','10','11','12','14','15'])
multirecording = multirecording.split_by('group')[1]

w = sw.plot_timeseries(multirecording,time_range=[0,15])


"""Pre Processing"""
#Bandpass filter
recording_f = spre.bandpass_filter(multirecording, freq_min=freq_min, freq_max=freq_max)
w = sw.plot_timeseries(recording_f,time_range=[10,15], segment_index=0)


#Median common ref
recording_cmr = spre.common_reference(recording_f, reference='global', operator='median')
w = sw.plot_timeseries(recording_cmr,time_range=[10,15], segment_index=0)
