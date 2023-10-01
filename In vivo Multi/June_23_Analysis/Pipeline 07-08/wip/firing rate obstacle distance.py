# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 17:38:02 2023

@author: Gilles.DELBECQ
"""

import pandas as pd
import matplotlib.pyplot as plt

# Lecture des données
# mocap_data = pd.read_excel(r"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0022_01_22_mocap.xlsx")
# rates_data = pd.read_excel(r"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0022_01_22_rates.xlsx")

# # Fusion des deux ensembles de données sur l'axe temporel
# merged_data = pd.merge(mocap_data, rates_data, on='time_axis', how='inner')

# # Liste des unités à tracer (exclut la colonne 'time_axis')
# # units_to_plot = rates_data.columns[1:]

# # Création d'une figure pour chaque unité
# for unit in units_to_plot:
#     plt.figure(figsize=(10, 5))
#     plt.scatter(merged_data["distance_from_obstacle_x"], merged_data[unit], s=2, label=unit)
#     plt.title(f"Firing Rate of {unit} vs. Distance from Obstacle X")
#     plt.xlabel("Distance from Obstacle X")
#     plt.ylabel("Firing Rate")
#     plt.legend()
#     plt.tight_layout()


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Listes des fichiers pour chaque session
rates_files = [
"E:/Data/ephy/Spikesorting August 2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0022_01_3_rates.xlsx",
"E:/Data/ephy/Spikesorting August 2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0022_01_4_rates.xlsx",
"E:/Data/ephy/Spikesorting August 2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0022_01_5_rates.xlsx",
"E:/Data/ephy/Spikesorting August 2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0022_01_6_rates.xlsx",
"E:/Data/ephy/Spikesorting August 2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0022_01_7_rates.xlsx"
    ]

mocap_files = [
"E:/Data/ephy/Spikesorting August 2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0022_01_3_mocap.xlsx",
"E:/Data/ephy/Spikesorting August 2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0022_01_4_mocap.xlsx",
"E:/Data/ephy/Spikesorting August 2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0022_01_5_mocap.xlsx",
"E:/Data/ephy/Spikesorting August 2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0022_01_6_mocap.xlsx",
"E:/Data/ephy/Spikesorting August 2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/0022_01_7_mocap.xlsx",
]
    
assert len(rates_files) == len(mocap_files), "Les listes rates_files et mocap_files doivent avoir la même longueur."

all_data = []

# Charger et fusionner les données pour chaque session
for rates_file, mocap_file in zip(rates_files, mocap_files):
    rates_data = pd.read_excel(rates_file)
    mocap_data = pd.read_excel(mocap_file)
    merged_data = pd.merge(mocap_data, rates_data, on='time_axis', how='inner')
    all_data.append(merged_data)

# Créer un DataFrame global pour toutes les sessions
global_data = pd.concat(all_data, ignore_index=True)

units_to_plot = rates_data.columns[1:]

marker_x = 'back1_x'

# Tracer pour chaque unité
for unit in units_to_plot:
    plt.figure(figsize=(10, 5))
    
    # Tracer les données de chaque session en semi-transparence
    test = []
    for data in all_data:
        # plt.plot(data[marker_x], data[unit], alpha=0.3)
        plt.scatter(data[marker_x], data[unit], alpha=0.3)
    
    # Calculer et tracer le taux de tir moyen sur toutes les sessions
    mean_rate = global_data.groupby(marker_x)[unit].mean()
    window_size = 10
    smoothed_mean_rate = mean_rate.rolling(window=window_size, center=True).mean()
    
    # plt.plot(mean_rate.index, smoothed_mean_rate.values, color='black', linewidth=2, label='Mean firing rate')
    

    plt.axvline(0)
    
    plt.title(f"Firing Rate of {unit} vs. Distance from Obstacle X")
    plt.xlabel("Distance from Obstacle X")
    plt.ylabel("Firing Rate")
    plt.legend()
    plt.tight_layout()
    plt.gca().invert_xaxis()  # Inversion de l'axe des x
    
