# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:01:45 2019

@author: gille
"""

from matplotlib import pyplot as plt 
import pandas as pd 
import numpy as np
import os
from scipy import stats 
import math
import glob

"""Détecter tous les spikes de la baseline
exclure les spikes suivant le premier spike d'un train
scatter plot
"""

Datapath1 = 'D:/Analyses/Baseline/SpikeSansTrain/'

seuil = 10 #ms

List_File_paths1 = []
for r, d, f in os.walk(Datapath1):
# r=root, d=directories, f = files
    for filename in f:
        if '.xls' in filename:
            List_File_paths1.append(os.path.join(r, filename))

           
for file_path in List_File_paths1:    
    CRITA = []
    CRITB = []
    SPIKE_TIME_N = []
    Spikes_time = np.array(pd.read_excel(file_path, usecols='B')*1000) #ms
    CritA = np.array(pd.read_excel(file_path, usecols='C')*1000) #ms
    CritB = np.array(pd.read_excel(file_path, usecols='D')*1000) #ms
    n = 0
    critA_n = np.array(CritA[n])
    critB_n = np.array(CritB[n])
    CRITA.append(critA_n)
    CRITB.append(critB_n)
    SPIKE_TIME_N.append(Spikes_time[n])
    n = 1
    print(file_path)
    for n in range(len(Spikes_time)):
        spike_n = Spikes_time[n]
        if spike_n >= Spikes_time[n-1]+seuil:
            critA_n = np.array(CritA[n])
            critB_n = np.array(CritB[n])
            CRITA.append(critA_n)
            CRITB.append(critB_n)
            SPIKE_TIME_N.append(spike_n)
    SPIKE_TIME_N = np.array(SPIKE_TIME_N).flatten()
    CRITA = np.array(CRITA).flatten()
    CRITB = np.array(CRITB).flatten()
    data_critere_spike = { 'Spike_time' : SPIKE_TIME_N, 'CritA' : CRITA, 'CritB' : CRITB}
    
    """A remplcare par un array? Array 1 : Spike time, array 2 CRITA, array 3 critB"""
    df5 = pd.DataFrame(data_critere_spike)
    df5.to_excel('D:/Analyses/Baseline/Data SpikeSansTrain/{}_sans_train.xlsx'.format(os.path.splitext(os.path.basename(file_path))[0]))
