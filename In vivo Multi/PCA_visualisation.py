# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:31:37 2022

@author: Gilles.DELBECQ
"""


import numpy as np
import matplotlib.pyplot as plt
import os, re
from scipy import stats
import scipy.signal as sp
import pandas as pd

"""
PARAMETERS
"""

sampling_rate = 20000


"""
Load excel file with waveforms and cluster #
"""
data_df=pd.read_excel('//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/Analysis/PCA test/PCA_results.xlsx', index_col=0)  

"""
List clusters
"""
clusters_idx = data_df.iloc[:, -1].unique()
clusters_idx = clusters_idx[~np.isnan(clusters_idx)]
"""
For each cluster : 
    - plot superposed waveform on 1 figure
    - plot mean trace
"""
mean_wvf=[]

for cluster in clusters_idx:
    cluster_data = data_df.loc[data_df.iloc[:, -1] == cluster].iloc[:,:-1].to_numpy()
    time_vector = np.arange(0,len(cluster_data[0])/sampling_rate,1/sampling_rate)*1000
    plt.figure()
    plt.title(cluster)
    for wvf in cluster_data:
        plt.plot(time_vector,wvf)
    
    mean_wvf.append(np.median(cluster_data, axis=0))

plt.figure()
for wvf in mean_wvf:
    plt.plot(time_vector,wvf)
    
