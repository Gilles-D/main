# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 17:58:52 2022

@author: Gilles.DELBECQ
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import cm

import os, re
from scipy import stats
import scipy.signal as sp
import pandas as pd

import seaborn as sns


"""
PARAMETERS
"""

sampling_rate = 20000


"""
Load excel file with waveforms and cluster #
"""
clusters=pd.read_excel('//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/Analysis/PCA test/PCA_results.xlsx', index_col=0).iloc[:, -1].to_numpy() 
parameters=pd.read_excel('//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/Analysis/PCA test/parameters_chan3.xlsx', index_col=0).to_numpy() 

parameters = pd.DataFrame(np.column_stack((parameters,clusters)))
"""
List clusters
"""
clusters_idx = parameters.iloc[:, -1].unique()
clusters_idx = clusters_idx[~np.isnan(clusters_idx)]

color = iter(cm.rainbow(np.linspace(0, 1, len(clusters_idx))))

plt.figure()
plt.title("Parameters")

for cluster in clusters_idx:
    c = next(color)
    cluster_data = parameters.loc[parameters.iloc[:, -1] == cluster].iloc[:,:-1].to_numpy()
    plt.scatter(cluster_data[:,0],cluster_data[:,2],c=c)
    