# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:34:30 2023

@author: Gilles.DELBECQ
"""

import numpy as np
import matplotlib.pyplot as plt

#Exported from tridesclou export (by spikeinterface to phy)
spike_times = np.load(r'//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/July_23/2_SI_data/0012_03_07_v2/phy_export/tridesclous/spike_times.npy')
spike_cluster = np.load(r'\\equipe2-nas1\Gilles.DELBECQ\Data\ePhy\July_23\2_SI_data/0012_03_07_v2/phy_export/tridesclous/spike_clusters.npy')
spike_templates = np.load(r'\\equipe2-nas1\Gilles.DELBECQ\Data\ePhy\July_23\2_SI_data/0012_03_07_v2/phy_export/tridesclous/similar_templates.npy')

sampling_rate=20000

#Get spikes for each cluster 
clusters_idx = np.unique(spike_cluster)#Index of all clusters

timings_to_exclude=[(0,359.5712),(1455.7888,1923.808),(670.5024,680.5024)]

# timings_to_exclude=[(359.5712,1923.808)]

#%%Cluster spikes loading
clustered_spike_times,clustered_spike_indexes=[],[]
for cluster in clusters_idx:        #Loop on each cluster to get the spike of the cluster
    array_idx = np.where(spike_cluster==cluster)[0]
    selected_spike_idx = np.take(spike_times,array_idx)#Spikes from the cluster
    
    
    selected_spike_times=selected_spike_idx/sampling_rate

    for i in timings_to_exclude:
        mask = np.logical_or(selected_spike_times <= i[0], selected_spike_times >= i[1])
        selected_spike_times = selected_spike_times[mask]

    clustered_spike_times.append(selected_spike_times) #All the spikes times in seconds by cluster

#%% All clusters correlogramm
for index,cluster in enumerate(clustered_spike_times):
    event_times = cluster*1000
    
    # Paramètres pour l'autocorrelogramme
    bin_size = 1  # Taille des intervalles de temps
    max_lag = 50  # Durée maximale de la corrélation
    

    # Calcul des différences entre les temps d'événements consécutifs
    event_diffs = np.diff(event_times)
    
    symetric_event_diffs = np.hstack((-event_diffs,event_diffs))
    
    # Création des bins pour l'autocorrelogramme
    bins = np.arange(-max_lag, max_lag + bin_size, bin_size)
    
    # Calcul de l'autocorrelogramme
    autocorrelogram, _ = np.histogram(symetric_event_diffs, bins=bins)
    
    # Affichage de l'autocorrelogramme
    plt.figure()
    plt.bar(bins[:-1], autocorrelogram, width=bin_size, align='edge')
    plt.xlabel('time in ms')
    plt.ylabel('Events')
    plt.title(rf'Unit # {index+1}')
    plt.show()
    plt.savefig(rf'\\equipe2-nas1\Gilles.DELBECQ\Data\ePhy\July_23\test\correlo_record1_{index+1}.png')

#%% Clusters to exclude
clusters_to_exclude = [4,5,6,10,12]

import numpy as np

# Exemple d'un tableau 1D
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Limites
lower_limit = 3
upper_limit = 7

# Création du masque booléen
mask = np.logical_and(arr > lower_limit, arr < upper_limit)

# Exclusion des valeurs comprises entre les limites
filtered_arr = arr[~mask]

# Affichage du tableau filtré
print(filtered_arr)
