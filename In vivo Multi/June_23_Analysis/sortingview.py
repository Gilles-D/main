# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:05:31 2024

@author: MOCAP
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:48:33 2024

@author: Gilles.DELBECQ
"""

#%% Imports and functions
import spikeinterface as si
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.exporters as sexp
import spikeinterface.widgets as sw
import spikeinterface.extractors as se
from spikeinterface.curation import MergeUnitsSorting

import os


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


import seaborn as sns

#%%
import kachery_cloud as kcl
kcl.init()

#%%

directory_list =[
    r'D:/ePhy/SI_Data/spikesorting_results/0030_01_09/kilosort3',
    r'D:/ePhy/SI_Data/spikesorting_results/0022_01_08/kilosort3'
    ]

#%% 
"""
Loading
"""
for sorter_folder in directory_list:
    
    
    sorter_result = se.NpzSortingExtractor.load_from_folder(rf'{sorter_folder}/in_container_sorting')
    we = si.WaveformExtractor.load_from_folder(f'{sorter_folder}\we')
    similarity = np.load(rf"{sorter_folder}\we\similarity\similarity.npy")
    units_location = spost.compute_unit_locations(we)
    correlograms = spost.compute_correlograms(we)
    
    plots = sw.plot_unit_waveforms(we, figsize=(16, 4))
    
    
    
    sw.plot_sorting_summary(waveform_extractor=we, curation=True, backend="sortingview")
