# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 13:56:20 2023

@author: Gilles.DELBECQ
"""

import pandas as pd

# 1. Charger le fichier Excel dans un DataFrame
df = pd.read_excel("//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/Mocap_data.xlsx")

# 2. Filtrer le DataFrame pour garder les lignes où au moins une colonne (à l'exception de 'time_axis' et 'Unnamed: 0') contient des données
df_filtered = df[df.drop(columns=['time_axis', 'Unnamed: 0']).notna().any(axis=1)]

df_filtered.to_excel("//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/Mocap_data_no_nan.xlsx")