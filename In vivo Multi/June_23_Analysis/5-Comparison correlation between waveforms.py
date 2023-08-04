# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 18:23:11 2023

@author: MOCAP
"""

import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import os

#%% Comparison correlation in one same list of waveforms

file_list =[
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_5 (Org_id_8_comp).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_8 (Org_id_14_moun).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_10 (Org_id_16_moun).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_14 (Org_id_20_moun).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_15 (Org_id_22_comp).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_16 (Org_id_12_spykingcircus).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_19 (Org_id_29_moun).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_20 (Org_id_30_comp).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_25 (Org_id_38_moun).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_29 (Org_id_48_moun).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_33 (Org_id_15_spykingcircus).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_45 (Org_id_6_spykingcircus).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_1 (Org_id_1_comp).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_4 (Org_id_5_comp).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_7 (Org_id_10_comp).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_11 (Org_id_15_moun).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_14 (Org_id_18_comp).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_18 (Org_id_23_comp).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_19 (Org_id_24_comp).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_21 (Org_id_28_moun).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_24 (Org_id_11_spykingcircus).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_25 (Org_id_24_spykingcircus).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_27 (Org_id_0_tdc).xlsx"
          ]



# Charger les données à partir des 50 fichiers CSV et stocker les waveforms dans une liste
waveforms = []
waveform_names = []
for i,file in enumerate(file_list):
    file_path = file  # Remplacez par les noms de vos fichiers CSV
    df = pd.read_excel(file_path)
    waveform = df.values[:, :].T
    waveforms.append(waveform)
    waveform_names.append(os.path.basename(file))

# Calculer les similarités cosinus entre les canaux pour chaque paire de waveforms
similarity_matrix = np.zeros((len(waveforms), len(waveforms)))
for i, waveform1 in enumerate(waveforms):
    for j, waveform2 in enumerate(waveforms):
        if j > i:
            similarity = 1 - cosine(waveform1.flatten(), waveform2.flatten())
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

# Calculer la similarité moyenne pour chaque waveform
mean_similarity = similarity_matrix.mean(axis=1)

# Trier les waveforms en fonction de leur similarité moyenne
sorted_indices = np.argsort(mean_similarity)[::-1]  # Tri décroissant

# Représenter graphiquement les similarités cosinus sous forme de heatmap
fig, ax = plt.subplots()
heatmap = ax.imshow(similarity_matrix[sorted_indices][:, sorted_indices], cmap='viridis', vmin=0, vmax=1)
plt.colorbar(heatmap, ax=ax)
ax.set_xticks(np.arange(len(waveforms)))
ax.set_yticks(np.arange(len(waveforms)))
ax.set_xticklabels(np.array(waveform_names)[sorted_indices], rotation=90)  # Utiliser les noms des fichiers triés
ax.set_yticklabels(np.array(waveform_names)[sorted_indices])
ax.set_xlabel('Waveform')
ax.set_ylabel('Waveform')
ax.set_title('Similarités cosinus entre les waveforms (triées par similarité moyenne)')

plt.show()

#%% Comparison between 2 lists of waveforms
file_list1 = [
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_1 (Org_id_1_comp).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_4 (Org_id_5_comp).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_7 (Org_id_10_comp).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_11 (Org_id_15_moun).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_14 (Org_id_18_comp).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_18 (Org_id_23_comp).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_19 (Org_id_24_comp).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_21 (Org_id_28_moun).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_24 (Org_id_11_spykingcircus).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_25 (Org_id_24_spykingcircus).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_27 (Org_id_0_tdc).xlsx"
]

file_list2 = [
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_45 (Org_id_6_spykingcircus).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_5 (Org_id_8_comp).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_8 (Org_id_14_moun).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_10 (Org_id_16_moun).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_14 (Org_id_20_moun).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_15 (Org_id_22_comp).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_16 (Org_id_12_spykingcircus).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_19 (Org_id_29_moun).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_20 (Org_id_30_comp).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_25 (Org_id_38_moun).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_29 (Org_id_48_moun).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_33 (Org_id_15_spykingcircus).xlsx"
]

# Charger les données à partir des fichiers CSV et stocker les waveforms dans des listes
waveforms1 = []
waveform_names1 = []
for file in file_list1:
    file_path = file
    df = pd.read_excel(file_path)
    waveform = df.values[:, 1:].T
    waveforms1.append(waveform)
    waveform_names1.append(os.path.basename(file))

waveforms2 = []
waveform_names2 = []
for file in file_list2:
    file_path = file
    df = pd.read_excel(file_path)
    waveform = df.values[:, 1:].T
    waveforms2.append(waveform)
    waveform_names2.append(os.path.basename(file))

# Calculer les similarités cosinus entre les canaux pour chaque paire de waveforms provenant des deux listes
similarity_matrix = np.zeros((len(waveforms1), len(waveforms2)))
for i, waveform1 in enumerate(waveforms1):
    for j, waveform2 in enumerate(waveforms2):
        similarity = 1 - cosine(waveform1.flatten(), waveform2.flatten())
        similarity_matrix[i, j] = similarity

# Représenter graphiquement les similarités cosinus sous forme de heatmap
fig, ax = plt.subplots()
heatmap = ax.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
plt.colorbar(heatmap, ax=ax)
ax.set_xticks(np.arange(len(waveforms2)))
ax.set_yticks(np.arange(len(waveforms1)))
ax.set_xticklabels(np.array(waveform_names2), rotation=90)  # Utiliser les noms des fichiers pour la liste 2
ax.set_yticklabels(np.array(waveform_names1))
ax.set_xlabel('Waveform (liste 2)')
ax.set_ylabel('Waveform (liste 1)')
ax.set_title('Similarités cosinus entre les waveforms des deux listes')
plt.tight_layout()
plt.show()
