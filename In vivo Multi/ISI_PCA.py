# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 11:06:50 2022

@author: Gilles.DELBECQ
"""
import numpy as np
import matplotlib.pyplot as plt
import os, re
from scipy import stats
import scipy.signal as sp
import pandas as pd

"""
Load spikes by cluster
for each spikes : position relative to other spikes
append it

plot a histogram

"""


"""
Load excel file with spike times and cluster #
"""
spike_times=pd.read_excel('//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/Analysis/PCA test/spike_times.xlsx', index_col=0).to_numpy().reshape(-1)

clusters=pd.read_excel('//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/Analysis/PCA test/test_waveform.xlsx', index_col=0).iloc[:, -1].to_numpy() 

"""
List clusters
"""
clusters_idx = np.unique(clusters)
clusters_idx = clusters_idx[~np.isnan(clusters_idx)]

spike_times = np.column_stack((spike_times*1000,clusters))  #in ms



for cluster in clusters_idx:
    spikes_clustered =spike_times[spike_times[:, 1] == cluster, :]
    print(rf'Cluster {cluster}')
    isi=[]

    for index,spike in np.ndenumerate(spikes_clustered[:,0]):
        if not index[0] == 0:
            interval = spike-spikes_clustered[index[0]-1,0]
            if interval < 10000 : isi.append(interval)
                       
    # for spike in spikes_clustered[:,0]:
    #     isi.append((spikes_clustered[:,0]-spike).tolist())
   
        
    
    bins=int((max(isi)-min(isi))/0.5)
    plt.figure()
    plt.title(rf'Cluster {cluster}')
    plt.hist(isi,bins=bins)
    plt.xlim(0,10)