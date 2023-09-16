# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 09:14:59 2023

@author: Gilles.DELBECQ
"""

import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#%% PCA on waveforms of all channels

# 1. Charger les données

all_files = [
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_75_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_2_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_5_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_6_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_7_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_8_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_13_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_16_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_20_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_21_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_23_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_34_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_38_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_39_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_40_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_42_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_51_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_57_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_59_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_60_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_62_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_64_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_65_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_67_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_70_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_71_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_72_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/waveforms/Unit_74_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_57_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_0_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_1_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_2_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_3_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_4_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_8_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_9_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_10_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_12_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_13_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_14_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_17_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_18_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_20_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_23_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_24_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_26_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_29_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_30_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_34_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_35_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_36_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_37_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_38_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_40_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_41_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_43_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_44_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_45_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_46_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_47_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_48_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_50_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_51_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_52_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_53_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_54_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_55_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/waveforms/Unit_56_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_47_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_0_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_1_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_2_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_3_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_4_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_5_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_6_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_7_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_9_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_10_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_11_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_12_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_13_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_14_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_15_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_17_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_18_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_19_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_20_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_23_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_25_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_27_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_28_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_29_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_30_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_32_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_34_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_37_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_38_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_39_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_41_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_44_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_45_wf.xlsx",
"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/waveforms/Unit_46_wf.xlsx"
]

def extract_name(filepath):
    parts = filepath.split("/")
    animal_id = parts[-6].split("_")[0]  # Extracting the animal id from the filepath
    unit_name = parts[-1].split('_')[1]  # Extracting the unit name from the filename
    return f"{animal_id}_Unit_{unit_name}"

neuron_names = [extract_name(file) for file in all_files]

all_concat_values = []
for file in all_files:
    data = pd.read_excel(file)
    data = data.drop(columns=data.columns[0])
    all_concat_values.append(data.values.flatten())

all_neurons_concat_df = pd.DataFrame(all_concat_values)

# 2. Standardisation des données

mean_all = all_neurons_concat_df.mean(axis=0)
std_all = all_neurons_concat_df.std(axis=0)
all_neurons_standardized = (all_neurons_concat_df - mean_all) / std_all

# 3. PCA

pca_all = PCA()
pca_result_all = pca_all.fit_transform(all_neurons_standardized)

# 4. Classification hiérarchique

n_clusters = 4

pca_data_26_components = pca_result_all[:, :26]
linked_all = linkage(pca_data_26_components, 'ward')
threshold = linked_all[-n_clusters+1, 2]

plt.figure(figsize=(15, 7))
dendrogram(linked_all, orientation='top', labels=neuron_names, distance_sort='descending', show_leaf_counts=True,color_threshold = threshold)
plt.title('Dendrogramme de classification hiérarchique')
plt.xlabel('Neurones')
plt.ylabel('Distance Ward')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Méthode du coude

wcss = []
max_clusters = 15
for i in range(1, max_clusters+1):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(pca_data_26_components)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_clusters+1), wcss, marker='o', linestyle='--')
plt.title('Méthode du coude (Elbow Method)')
plt.xlabel('Nombre de clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. k-means avec 2 clusters
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(pca_data_26_components)
plt.figure(figsize=(15, 10))
for i in range(n_clusters):
    plt.scatter(pca_result_all[clusters == i, 0], pca_result_all[clusters == i, 1], label=f'Cluster {i+1}', s=100)
for i, neuron_name in enumerate(neuron_names):
    plt.annotate(neuron_name, (pca_result_all[i, 0], pca_result_all[i, 1]))
plt.xlabel('Premier composant principal')
plt.ylabel('Deuxième composant principal')
plt.title('Répartition des neurones avec k-means')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#%% Extract largest amplitude

# Function to extract the waveform of the channel with the largest amplitude
def extract_largest_waveform(data):
    # Calculate the amplitude for each channel
    amplitude = data.max(axis=0) - data.min(axis=0)
    # Get the channel with the largest amplitude
    largest_channel = amplitude.idxmax()
    return data[largest_channel].values

largest_waveforms = []

# Extract the waveform with the largest amplitude for each neuron and append to the list
for file in all_files:
    data = pd.read_excel(file).set_index('Unnamed: 0')
    largest_waveforms.append(extract_largest_waveform(data))

# # Plot the largest waveforms for each neuron
# plt.figure(figsize=(15, 10))
# for i, waveform in enumerate(largest_waveforms):
#     plt.plot(waveform, label=neuron_names[i], alpha=0.7)

# plt.title('Waveforms du canal avec la plus grande amplitude pour chaque neurone')
# plt.xlabel('Échantillons temporels')
# plt.ylabel('Amplitude')
# plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
# plt.grid(True)
# plt.tight_layout()
# plt.show()

def center_waveform_on_min(waveform):
    """Center the waveform on its minimum value."""
    min_idx = np.argmin(waveform)
    center_idx = len(waveform) // 2
    shift = center_idx - min_idx
    centered_waveform = np.roll(waveform, shift)
    return centered_waveform

# Center each waveform on its minimum value
centered_waveforms = [center_waveform_on_min(wf) for wf in largest_waveforms]

# Plot the centered waveforms
plt.figure(figsize=(15, 10))
for i, waveform in enumerate(centered_waveforms):
    plt.plot(waveform, label=neuron_names[i], alpha=0.7)

plt.title('Waveforms centrées sur leur amplitude négative maximale')
plt.xlabel('Échantillons temporels')
plt.ylabel('Amplitude')
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.grid(True)
plt.tight_layout()
plt.show()

def extract_all_waveform_features_adjusted(waveform):
    """Extract all action potential waveform features with adjusted amplitude."""
    
    # Find the minimum value (trough) and its index
    min_val = np.min(waveform)
    min_idx = np.argmin(waveform)
    
    # Amplitude defined as the absolute value of the trough
    amplitude = abs(min_val)
    
    # Find the maximum value (peak) after the trough and its index
    post_min_vals = waveform[min_idx:]
    post_min_peak_val = np.max(post_min_vals)
    post_min_peak_idx = np.argmax(post_min_vals) + min_idx
    
    # Find the maximum value (peak) before the trough and its index
    pre_min_vals = waveform[:min_idx]
    pre_min_peak_val = np.max(pre_min_vals)
    pre_min_peak_idx = np.argmax(pre_min_vals)
    
    # Correctly calculate half-width
    half_amplitude = min_val / 2
    left_half_idx = np.where(waveform[:min_idx] > half_amplitude)[0][-1]
    right_half_idx = np.where(waveform[min_idx:] > half_amplitude)[0][0] + min_idx
    half_width = right_half_idx - left_half_idx
    
    # Duration from start to min
    # Updated calculation for Time to Min
    time_to_min = min_idx - pre_min_peak_idx
    
    time_to_min_peak = min_idx
    
    # Duration from start to next peak
    time_to_next_peak = post_min_peak_idx
    
    # Action potential duration (from min to next peak)
    ap_duration = post_min_peak_idx - min_idx
    
    # Find the index where the waveform returns to baseline after the peak
    baseline_return_idx = np.where(post_min_vals[post_min_peak_idx - min_idx:] > 0)[0]
    
    # If waveform doesn't return to baseline, set the end of waveform as the return point
    if len(baseline_return_idx) == 0:
        baseline_return_idx = len(waveform) - 1
    else:
        baseline_return_idx = baseline_return_idx[0] + post_min_peak_idx

    # Calculate recovery duration
    recovery_duration = baseline_return_idx - post_min_peak_idx
    
    # Calculate recovery slope
    recovery_slope = (0 - post_min_peak_val) / recovery_duration
    
    # Calculate repolarization slope
    repolarization_slope = (post_min_peak_val - min_val) / (post_min_peak_idx - min_idx)
    
    return {
        "Amplitude": amplitude,
        "Half Width": half_width,
        "AP Duration": ap_duration,
        "Time to Min": time_to_min,
        "Time to Next Peak": time_to_next_peak,
        "Recovery Duration": recovery_duration,
        "Recovery Slope": recovery_slope,
        "Repolarization Slope": repolarization_slope,
        "time_to_min_peak" : time_to_min_peak
    }

# Extract all waveform features for each neuron with adjusted amplitude
all_waveform_features_adjusted = {neuron: extract_all_waveform_features_adjusted(wf) for neuron, wf in zip(neuron_names, centered_waveforms)}

all_waveform_features_adjusted_df = pd.DataFrame(all_waveform_features_adjusted).T
all_waveform_features_adjusted_df


#%% Example waveform parameters
# Selecting a representative waveform from the available neurons
example_neuron = neuron_names[0]  # Choosing the first neuron for demonstration
example_waveform = centered_waveforms[neuron_names.index(example_neuron)]
features = extract_all_waveform_features_adjusted(example_waveform)

# Plot the waveform
plt.figure(figsize=(15, 8))
plt.plot(example_waveform, label="Waveform", color="blue", lw=2)

# Highlight the amplitude
plt.plot([features["Time to Min"], features["Time to Min"]], [0, -features["Amplitude"]], color="red", linestyle="--", label="Amplitude")
plt.scatter(features["Time to Min"], -features["Amplitude"], color="red", s=50, zorder=5)

# Highlight the half width
plt.axhline(y=-features["Amplitude"] / 2, color="green", linestyle="--", label="Half Amplitude")
plt.scatter([features["Time to Min"] - features["Half Width"] / 2, features["Time to Min"] + features["Half Width"] / 2], 
            [-features["Amplitude"] / 2, -features["Amplitude"] / 2], color="green", s=50, zorder=5)

# Highlight the AP duration
plt.plot([features["Time to Min"], features["Time to Next Peak"]], [-features["Amplitude"], example_waveform[features["Time to Next Peak"]]], 
         color="purple", linestyle="--", label="AP Duration")
plt.scatter(features["Time to Next Peak"], example_waveform[features["Time to Next Peak"]], color="purple", s=50, zorder=5)

# Highlight the recovery duration and slope
if not np.isinf(features["Recovery Slope"]):
    plt.plot([features["Time to Next Peak"], features["Time to Next Peak"] + features["Recovery Duration"]],
             [example_waveform[features["Time to Next Peak"]], 0], color="orange", linestyle="--", label="Recovery Duration & Slope")
    plt.scatter(features["Time to Next Peak"] + features["Recovery Duration"], 0, color="orange", s=50, zorder=5)

# Highlight the repolarization slope
plt.plot([features["Time to Min"], features["Time to Next Peak"]], [-features["Amplitude"], example_waveform[features["Time to Next Peak"]]], 
         color="cyan", linestyle="--", label="Repolarization Slope")

plt.title(f"Caractéristiques du potentiel d'action pour le neurone {example_neuron}")
plt.xlabel("Échantillons temporels")
plt.ylabel("Amplitude")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()


#%%

