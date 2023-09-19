# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:19:47 2023

@author: Gilles.DELBECQ
"""
#%% Imports and functions
import spikeinterface as si
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.exporters as sexp
import spikeinterface.widgets as sw
from spikeinterface.curation import MergeUnitsSorting
import spikeinterface.comparison as sc

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

#%%Parameters
sorter_name='kilosort3'

concatenated_signals_path = r'D:\ePhy\SI_Data\concatenated_signals'
spikesorting_results_folder =r"D:\ePhy\SI_Data\spikesorting_results"


session1 = '0022_01_08'
session2 = '0022_07_08'

"""
Loading
"""

sorter_name = "kilosort3"

sorter_folder1 = rf'{spikesorting_results_folder}/{session1}/{sorter_name}/curated'
signal_folder1 = rf'{concatenated_signals_path}/{session1}'

sorter_folder2 = rf'{spikesorting_results_folder}/{session2}/{sorter_name}/curated'
signal_folder2 = rf'{concatenated_signals_path}/{session2}'

sorter_result_session1 = ss.NpzSortingExtractor.load_from_folder(r'D:\ePhy\SI_Data\spikesorting_results\0022_01_08\kilosort3\in_container_sorting')
sorter_result_session2 = ss.NpzSortingExtractor.load_from_folder(r'D:\ePhy\SI_Data\spikesorting_results\0022_07_08\kilosort3\in_container_sorting')


we_session1 = si.WaveformExtractor.load_from_folder(rf'{sorter_folder1}/waveforms')
we_session2 = si.WaveformExtractor.load_from_folder(f'{sorter_folder2}\waveforms')

p_tcmp = sc.compare_templates(we_session1,we_session2)
