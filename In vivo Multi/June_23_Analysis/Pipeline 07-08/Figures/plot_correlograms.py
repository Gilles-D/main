# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 15:34:07 2023

@author: Gil
"""

#%% Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spikeinterface as si
import spikeinterface.sorters as ss
import spikeinterface.widgets as sw
import pickle



#%%Parameters
session_name = '0022_01_08'

spikesorting_results_path = r"D:\ePhy\SI_Data\spikesorting_results"
concatenated_signals_path = r'D:\ePhy\SI_Data\concatenated_signals'
plots_path = r'D:\ePhy\SI_Data\plots\correlograms'

sorter_name = "kilosort3"

sorter_folder = rf'{spikesorting_results_path}/{session_name}/{sorter_name}'
signal_folder = rf'{concatenated_signals_path}/{session_name}'

sampling_rate = 20000 #Hz

#%% Functions

def Check_Save_Dir(save_path):
    """
    Check if the save folder exists. If not, create it.

    Args:
        save_path (str): Path to the save folder.

    """
    import os
    isExist = os.path.exists(save_path)
    if not isExist:
        os.makedirs(save_path)  # Create folder for the experiment if it does not already exist

    return

def Get_recordings_info(session_name, concatenated_signals_path, spikesorting_results_path):
    """
    Cette fonction récupère les informations d'enregistrement à partir d'un fichier de métadonnées
    dans le dossier de signaux concaténés.

    Args:
        session_name (str): Le nom de la session d'enregistrement.
        concatenated_signals_path (str): Le chemin vers le dossier contenant les signaux concaténés.
        spikesorting_results_path (str): Le chemin vers le dossier des résultats du tri des spikes.

    Returns:
        dict or None: Un dictionnaire contenant les métadonnées si la lecture est réussie,
        ou None si la lecture échoue.

    Raises:
        Exception: Si une erreur se produit pendant la lecture du fichier.

    """
    try:
        # Construire le chemin complet vers le fichier de métadonnées
        path = rf'{concatenated_signals_path}/{session_name}/'
        
        # Lire le fichier de métadonnées à l'aide de la bibliothèque pickle
        print("Lecture du fichier ttl_idx dans le dossier Intan...")
        metadata = pickle.load(open(rf"{path}/ttl_idx.pickle", "rb"))
        
    except Exception as e:
        # Gérer toute exception qui pourrait se produire pendant la lecture du fichier
        print("Aucune information d'enregistrement trouvée dans le dossier Intan. Veuillez exécuter l'étape 0.")
        metadata = None  # Aucune métadonnée disponible en cas d'erreur
    
    print('Terminé')
    return metadata

#%% Loadings
"""
Load units
"""
recordings_info = Get_recordings_info(session_name,concatenated_signals_path,spikesorting_results_path)

print(rf"Loading spikesorting results for session {session_name}")
sorter_results = ss.NpzSortingExtractor.load_from_folder(rf'{sorter_folder}/curated').remove_empty_units()
signal = si.load_extractor(signal_folder)
we = si.load_waveforms(rf'{sorter_folder}/curated/waveforms')

time_axis = signal.get_times()
unit_list = sorter_results.get_unit_ids()

print(rf"{len(sorter_results.get_unit_ids())} units loaded")

#%% Auto corr

for unit in unit_list:
    sw.plot_autocorrelograms(sorter_results, window_ms=150.0, bin_ms=1.0, unit_ids=np.array([unit]))
    Check_Save_Dir(plots_path)
    plt.savefig(rf"{plots_path}/Autocorr_unit_{unit}.png")
    plt.close()



#%% Cross corr

sw.plot_crosscorrelograms(sorter_results)

plt.savefig(rf"{plots_path}/Correlogram.png")
plt.close('all')


# from itertools import combinations

# # Supposons que unit_list contienne la liste de vos unités
# num_units = len(unit_list)

# # Divisez la liste d'unités en 4 groupes
# group_size = num_units // 6
# unit_groups = [unit_list[i:i + group_size] for i in range(0, num_units, group_size)]

# # Générer toutes les combinaisons possibles 2 à 2 des groupes
# combinations_2by2 = list(combinations(unit_groups, 2))

# # Afficher toutes les combinaisons
# for i,comb in enumerate(combinations_2by2):
#     sw.plot_crosscorrelograms(sorter_results, unit_ids = np.concatenate(comb))
    
#     plt.savefig(rf"{plots_path}/corr_{i}.png")
#     plt.close('all')
    
