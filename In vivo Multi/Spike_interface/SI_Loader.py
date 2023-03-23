# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:41:02 2023

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




ss.read_sorter_folder('//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Février2023/Test_Gustave/sorting_output/Gustave_22_03_baseline/kilosort3')