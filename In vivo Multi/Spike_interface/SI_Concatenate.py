# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 12:22:00 2023

@author: Gilles.DELBECQ
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
# probe_path=r'D:/ePhy/SI_Data/Buzsaki16.json'

# Data folder path
working_dir=r'D:\ePhy\Intan_Data'

# Saving Folder path
saving_dir=r"D:\ePhy\SI_Data"

subject_name="_7"

saving_name="_7_10_05_02"

freq_min = 300
freq_max = 6000

#Modifier en une boucle sur le dossier itan
recordings = [  
"D:/ePhy/Intan_Data/_7/05_10/_7_10_05_23_230510_173526/_7_10_05_23_230510_173526.rhd",
"D:/ePhy/Intan_Data/_7/05_10/_7_10_05_23_230510_173526/_7_10_05_23_230510_173626.rhd",
"D:/ePhy/Intan_Data/_7/05_10/_7_10_05_23_230510_173526/_7_10_05_23_230510_173726.rhd",
"D:/ePhy/Intan_Data/_7/05_10/_7_10_05_23_230510_173526/_7_10_05_23_230510_173826.rhd",
"D:/ePhy/Intan_Data/_7/05_10/_7_10_05_23_230510_173526/_7_10_05_23_230510_173926.rhd",
"D:/ePhy/Intan_Data/_7/05_10/_7_10_05_23_230510_173526/_7_10_05_23_230510_174026.rhd"]




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



#excluded_sites = ['2','3','4','9','10','11','12','14','15']
excluded_sites = []

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


"""
------------------Saving------------------
"""
job_kwargs = dict(n_jobs=1, chunk_duration="1s", progress_bar=False)
recording_saved = recording_cmr.save(folder=rf"{saving_dir}/{subject_name}/{saving_name}", **job_kwargs)
recording_saved


