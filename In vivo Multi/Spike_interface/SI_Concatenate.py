# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 12:22:00 2023

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
------------------PARAMETERS------------------

"""
probe_path=r'//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/A1x16-Poly2-5mm-50s-177.json'
# probe_path=r'//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Buzsaki16.json'

# Working folder path
working_dir=r'//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Février2023/Test_Gustave/raw/raw si/'

saving_name="Gust_27_02_baseline_fixed2"

freq_min = 300
freq_max = 6000


recordings = [  
"//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Février2023/Test_Gustave/raw/raw intan/Gustave_27_02_230227_162822/Gustave_27_02_230227_162822.rhd",
"//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Février2023/Test_Gustave/raw/raw intan/Gustave_27_02_230227_162822/Gustave_27_02_230227_162922.rhd",
"//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Février2023/Test_Gustave/raw/raw intan/Gustave_27_02_230227_162822/Gustave_27_02_230227_163022.rhd",
"//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Février2023/Test_Gustave/raw/raw intan/Gustave_27_02_230227_162822/Gustave_27_02_230227_163122.rhd",
"//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Février2023/Test_Gustave/raw/raw intan/Gustave_27_02_230227_162822/Gustave_27_02_230227_163222.rhd",
"//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Février2023/Test_Gustave/raw/raw intan/Gustave_27_02_230227_162822/Gustave_27_02_230227_163322.rhd",
"//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Février2023/Test_Gustave/raw/raw intan/Gustave_27_02_230227_163558/Gustave_27_02_230227_163558.rhd",
"//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Février2023/Test_Gustave/raw/raw intan/Gustave_27_02_230227_163558/Gustave_27_02_230227_163658.rhd",
"//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Février2023/Test_Gustave/raw/raw intan/Gustave_27_02_230227_163558/Gustave_27_02_230227_163758.rhd",
"//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Février2023/Test_Gustave/raw/raw intan/Gustave_27_02_230227_163558/Gustave_27_02_230227_163858.rhd",
"//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Février2023/Test_Gustave/raw/raw intan/Gustave_27_02_230227_163558/Gustave_27_02_230227_163958.rhd",
"//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Février2023/Test_Gustave/raw/raw intan/Gustave_27_02_230227_163558/Gustave_27_02_230227_164058.rhd",
"//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Février2023/Test_Gustave/raw/raw intan/Gustave_27_02_230227_164330/Gustave_27_02_230227_164330.rhd",
"//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Février2023/Test_Gustave/raw/raw intan/Gustave_27_02_230227_164330/Gustave_27_02_230227_164430.rhd",
"//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Février2023/Test_Gustave/raw/raw intan/Gustave_27_02_230227_164330/Gustave_27_02_230227_164530.rhd",
"//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Février2023/Test_Gustave/raw/raw intan/Gustave_27_02_230227_164330/Gustave_27_02_230227_164630.rhd",
"//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Février2023/Test_Gustave/raw/raw intan/Gustave_27_02_230227_164330/Gustave_27_02_230227_164730.rhd"




]




if os.path.exists(working_dir)==False:
    os.makedirs(working_dir)
os.chdir(working_dir)



#Concatenate all the recording files
recordings_list=[]
for recording_file in recordings:
    recording = si.extractors.read_intan(recording_file,stream_id='0')
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

excluded_sites = ['2','3','4','9','10','11','12','14','15']


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
------------------Saving------------------
"""
job_kwargs = dict(n_jobs=4, chunk_duration="1s", progress_bar=False)
recording_saved = recording_cmr.save(folder=rf"{working_dir}/{saving_name}", **job_kwargs)
recording_saved


