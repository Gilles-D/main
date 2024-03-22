# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:34:21 2024

@author: MOCAP
"""

#%% modules
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
import sys
import time

import probeinterface as pi
from probeinterface.plotting import plot_probe

import warnings
warnings.simplefilter("ignore")


import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


from viziphant.statistics import plot_time_histogram
from viziphant.rasterplot import rasterplot_rates
from elephant.statistics import time_histogram
from neo.core import SpikeTrain
from quantities import s, ms
import pandas as pd

#%%Parameters
#Folder containing the folders of the session
animal_id = "0030"
session_name = "0030_01_09"
saving_name=session_name

rhd_folder = rf'D:\ePhy\Intan_Data\{animal_id}\{session_name}'


probe_path=r'D:/ePhy/SI_Data/A1x16-Poly2-5mm-50s-177.json'   #INTAN Optrode

# Sites to exclude
excluded_sites = []


def list_recording_files(path):
    """
    List all recording files (.rhd) in the specified directory and its subdirectories.
    
    Parameters:
        path (str): The directory path to search for recording files.
        
    Returns:
        list: A list of paths to all recording files found.
    """
    
    import glob
    fichiers = [fichier for fichier in glob.iglob(path + '/**/*', recursive=True) if not os.path.isdir(fichier) and fichier.endswith('.rhd')]
    
    return fichiers


#%%Main script


recordings = list_recording_files(rhd_folder)

"""------------------Concatenation------------------"""
recordings_list=[]
for recording_file in recordings:
    recording = se.read_intan(recording_file,stream_id='0')
    recording.annotate(is_filtered=False)
    recordings_list.append(recording)

multirecording = si.concatenate_recordings(recordings_list)


recording_info_path = os.path.dirname((os.path.dirname(recordings[0])))
recording_info = pickle.load(open(rf'{recording_info_path}/ttl_idx.pickle', "rb"))

         

"""------------------Set the probe------------------"""
probe = pi.io.read_probeinterface(probe_path)
probe = probe.probes[0]
multirecording = multirecording.set_probe(probe)
plot_probe(probe, with_device_index=True)

time_rage=[598,600]
# channels = ['7']

channels = multirecording.get_channel_ids()


"""------------------Defective sites exclusion------------------"""


stim_idx = recording_info['stim_ttl_on']
multirecording_arte = spre.remove_artifacts(multirecording,stim_idx, ms_before=1.2, ms_after=1.2,mode='cubic')


    

"""------------------Pre Processing------------------"""
#Bandpass filter
recording_f = spre.bandpass_filter(multirecording_arte, freq_min=300, freq_max=6000)
recording_cmr_ = spre.common_reference(recording_f, reference='global', operator='median')
sw.plot_timeseries(recording_cmr_, channel_ids=channels,time_range=time_rage)


for i in [num for num in range(300, 6000 + 1) if num % 200 == 0]:
    recording_f = spre.notch_filter(recording_f, freq=i)

recording_cmr = spre.common_reference(recording_f, reference='global', operator='median')
sw.plot_timeseries(recording_cmr, channel_ids=channels,time_range=time_rage)




#%%

"""


recording_cmr = spre.common_reference(recording_f, reference='global', operator='median')



for i in [num for num in range(200, 6000 + 1) if num % 200 == 0]:
    recording_f_notched = spre.notch_filter(recording_f, freq=i,q=30)
    recording_f_MOCAP_Arte = spre.notch_filter(recording_f, freq=i,q=30)
    recording_cmr_MOCAP_Arte = spre.notch_filter(recording_cmr, freq=i,q=30)
    recording_MOCAP_Arte = spre.notch_filter(multirecording_arte, freq=i,dtype='int16',q=30)


recording_f_then_cmr_MOCAP_arte = spre.common_reference(recording_f_MOCAP_Arte, reference='global', operator='median')

recording_MOCAP_Arte_f = spre.bandpass_filter(recording_MOCAP_Arte, freq_min=300, freq_max=3000)
recording_MOCAP_Arte_cmr = spre.common_reference(recording_MOCAP_Arte_f, reference='global', operator='median')


recording_f_notched_cmr=spre.common_reference(recording_f_notched, reference='global', operator='average')
for i in [num for num in range(200, 6000 + 1) if num % 200 == 0]:
    recording_f_notched_cmr_notched = spre.notch_filter(recording_f_notched_cmr, freq=i,q=100)



sw.plot_timeseries(multirecording, channel_ids=channels,time_range=time_rage)
plt.title('Raw')
sw.plot_timeseries(multirecording_arte, channel_ids=channels,time_range=time_rage)
plt.title('Raw - Stim Artefact removed')
sw.plot_timeseries(recording_f, channel_ids=channels,time_range=time_rage)
plt.title('filtered')
sw.plot_timeseries(recording_cmr, channel_ids=channels,time_range=time_rage)
plt.title('CMR')

sw.plot_timeseries(recording_cmr, channel_ids=channels,time_range=time_rage)
plt.title('Filtered - CMR - Artefact removed')

sw.plot_timeseries(recording_f_then_cmr_MOCAP_arte, channel_ids=channels,time_range=time_rage)
plt.title('Filtered - MOCAP Artefact removed - CMR')

sw.plot_timeseries(recording_cmr_MOCAP_Arte, channel_ids=channels,time_range=time_rage)
plt.title('Filtered - CMR - MOCAP Artefact removed')

sw.plot_timeseries(recording_f_notched, channel_ids=channels,time_range=time_rage)
plt.title('Filtered MOCAP Artefact removed')

sw.plot_timeseries(recording_f_notched_cmr, channel_ids=channels,time_range=time_rage)
plt.title('Filtered MOCAP Artefact removed CMR')

# sw.plot_timeseries(recording_MOCAP_Arte_cmr, channel_ids=channels,time_range=time_rage)
# plt.title('MOCAP Artefact removed - Filtered - CMR - ')


"""
#%%
recording = si.load_extractor('D:/ePhy/SI_Data/concatenated_signals/0030_01_09')

sw.plot_timeseries(recording, channel_ids=channels,time_range=time_rage)



#%%
recording_f = spre.bandpass_filter(multirecording_arte, freq_min=300, freq_max=6000)

for i in [num for num in range(300, 6000 + 1) if num % 200 == 0]:
    recording_f = spre.notch_filter(recording_f, freq=i)

recording_cmr = spre.common_reference(recording_f, reference='global', operator='median')

sw.plot_timeseries(recording_cmr, channel_ids=channels,time_range=time_rage)
