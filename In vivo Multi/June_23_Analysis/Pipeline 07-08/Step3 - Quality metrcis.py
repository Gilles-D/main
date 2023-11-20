# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 10:07:55 2023

@author: Gilles Delbecq

Perform automatic and manual curation over a sorter result (spikeinterface sorter result - for one sorter here)

Inputs : Spike sorting results (for instance from a session)

- performs automatic curation (with quality metrics parameters)
- performs manual curation (with manual selection/split/merge of units)
- compute metrics for units (waveform, location)

Returns: a curated sorter result (spikeinterface) and computed metrics, and waveforms

"""

#%% Imports and functions
import spikeinterface as si
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.exporters as sexp
import spikeinterface.widgets as sw
from spikeinterface.curation import MergeUnitsSorting

import os


import warnings
warnings.simplefilter("ignore")


import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from viziphant.statistics import plot_time_histogram
from viziphant.rasterplot import rasterplot_rates
from elephant.statistics import time_histogram
from neo.core import SpikeTrain
from quantities import s, ms

import pandas as pd


import seaborn as sns


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

def list_curated_units(directory):
    units = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            unit_id = file_name.split('_')[1].split('(')[0].split(' ')[0]
            unit_org_id = file_name.split('_')[3]
            unit_sorter = file_name.split('_')[-1].split(')')[0]
            units.append((unit_id,unit_org_id,unit_sorter))
    return units

def get_similarity_couples(similarity,similarity_threshold):
    row_indices, col_indices = np.where(similarity > similarity_threshold)
    # Utiliser un ensemble pour stocker les couples d'unités uniques
    unique_couples = set()

    # Parcourir les indices et ajouter les couples d'unités uniques à l'ensemble
    for row_idx, col_idx in zip(row_indices, col_indices):
        if row_idx != col_idx:
            unique_couples.add((min(row_idx, col_idx), max(row_idx, col_idx)))

    # Convertir l'ensemble en liste de couples d'unités
    couples_unites = list(unique_couples)
    
    return couples_unites


def plot_maker(sorter, we,unit_list):
    """
    Generate and save plots for an individual sorter's results.
    
    Parameters:
        sorter (spikeinterface.SortingExtractor): The sorting extractor containing the results of a spike sorter.
        we (spikeinterface.WaveformExtractor): The waveform extractor for the sorting extractor.
        save (bool): Whether to save the generated plots.
        sorter_name (str): Name of the spike sorter.
        save_path (str): Directory where the plots will be saved.
        saving_name (str): Name of the recording data.
        
    Returns:
        None
    """
    
    for unit_id in unit_list:
        fig = plt.figure(figsize=(25, 13))
        gs = GridSpec(nrows=3, ncols=6)
        fig.suptitle(f'{unit_id} (Total spike {sorter.get_total_num_spikes()[unit_id]})',)
        ax0 = fig.add_subplot(gs[0, 0:3])
        ax1 = fig.add_subplot(gs[0, 3:7])
        ax1.set_title('Mean firing rate during a trial')
        ax2 = fig.add_subplot(gs[1, :])
        ax2.set_title('Waveform of the unit')
        ax3 = fig.add_subplot(gs[2, 0])
        ax4 = fig.add_subplot(gs[2, 1], sharey = ax3)
        ax5 = fig.add_subplot(gs[2, 2], sharey = ax3)
        ax6 = fig.add_subplot(gs[2, 3:6])
        sw.plot_autocorrelograms(sorter, unit_ids=[unit_id], axes=ax0, bin_ms=1, window_ms=200)
        ax0.set_title('Autocorrelogram')
        current_spike_train = sorter.get_unit_spike_train(unit_id)/sorter.get_sampling_frequency()
        current_spike_train_list = []
        while len(current_spike_train) > 0: #this loop is to split the spike train into trials with correct duration in seconds
            # Find indices of elements under 9 (9 sec being the duration of the trial)
            indices = np.where(current_spike_train < 9)[0]
            if len(indices)>0:
                # Append elements to the result list
                current_spike_train_list.append(SpikeTrain(current_spike_train[indices]*s, t_stop=9))
                # Remove the appended elements from the array
                current_spike_train = np.delete(current_spike_train, indices)
                # Subtract 9 from all remaining elements
            current_spike_train -= 9
        bin_size = 100
        histogram = time_histogram(current_spike_train_list, bin_size=bin_size*ms, output='mean')
        histogram = histogram*(1000/bin_size)
        ax1.axvspan(0, 0.5, color='green', alpha=0.3)
        ax1.axvspan(1.5, 2, color='green', alpha=0.3)
        ax6.axvspan(0, 0.5, color='green', alpha=0.3)
        ax6.axvspan(1.5, 2, color='green', alpha=0.3)
        plot_time_histogram(histogram, units='s', axes=ax1)
        sw.plot_unit_waveforms_density_map(we, unit_ids=[unit_id], ax=ax2)
        template = we.get_template(unit_id=unit_id).copy()
        
        for curent_ax in [ax3, ax4, ax5]:
            max_channel = np.argmax(np.abs(template))%template.shape[1]
            template[:,max_channel] = 0
            mean_residual = np.mean(np.abs((we.get_waveforms(unit_id=unit_id)[:,:,max_channel] - we.get_template(unit_id=unit_id)[:,max_channel])), axis=0)
            curent_ax.plot(mean_residual)
            curent_ax.plot(we.get_template(unit_id=unit_id)[:,max_channel])
            curent_ax.set_title('Mean residual of the waveform for channel '+str(max_channel))
        plt.tight_layout()
        rasterplot_rates(current_spike_train_list, ax=ax6, histscale=0.1)

def fractionner_liste(liste, taille_sous_liste):
    sous_listes = []
    for i in range(0, len(liste), taille_sous_liste):
        sous_liste = liste[i:i + taille_sous_liste]
        sous_listes.append(sous_liste)
    return sous_listes


#%% Parameters
session_name = '0026_01_08'
sorter_name='kilosort3'
# sorter_name='mountainsort4'


concatenated_signals_path = r'D:\ePhy\SI_Data\concatenated_signals'
spikesorting_results_folder = r'D:\ePhy\SI_Data\spikesorting_results'
sorter_folder = rf'{spikesorting_results_folder}/{session_name}/{sorter_name}'
signal_folder = rf'{concatenated_signals_path}/{session_name}'




"""
https://spikeinterface.readthedocs.io/en/latest/modules/qualitymetrics.html

Criterium to exclude units :
    
    (- refractory period < 0.5% (1ms))
    - minimum frequency < 0.1 Hz
    - presence ratio > 90 (eliminate artefacts?)
    - ISI violation ratio > 5
    - L ratio > 10 ?
    
"""



max_isi = 5
min_frequency = 0.1
min_presence = 0.9
max_l_ratio = 10

similarity_threshold = 0.9


#%% One sorter auto-curation
"""
Loading
"""
sorter_result = ss.NpzSortingExtractor.load_from_folder(rf'{sorter_folder}/in_container_sorting')
we = si.WaveformExtractor.load_from_folder(f'{sorter_folder}\we')
similarity = np.load(rf"{sorter_folder}\we\similarity\similarity.npy")

#read signal (concatenated)
signal = si.load_extractor(signal_folder)


"""
Computing metrics
"""
# qm_params = sqm.get_default_qm_params()
# print(qm_params)

try:
    quality_metrics = sqm.compute_quality_metrics(we, load_if_exists=True)
except:
    quality_metrics = sqm.compute_quality_metrics(we, load_if_exists=False)
spike_id_list = quality_metrics.index

"""
Filtering with criteria
"""
# Filter the units with good quality metrics, according to the selected parameters
crit_ISI = quality_metrics['isi_violations_ratio'] < max_isi
crit_frequency = quality_metrics['firing_rate'] > min_frequency
crit_presence = quality_metrics['presence_ratio'] > min_presence
crit_l_ratio = quality_metrics['l_ratio'] < max_l_ratio

selected_quality_metrics = quality_metrics[crit_ISI & crit_frequency & crit_presence & crit_l_ratio]
selected_spike_id_list = selected_quality_metrics.index

spikes_not_passing_quality_metrics = list(set(spike_id_list) - set(selected_spike_id_list))

print(rf"{spikes_not_passing_quality_metrics} removed")

#%% Similarity computing
# Detect the units with high similarity and save them

similarity_couples = get_similarity_couples(similarity,similarity_threshold)
similarity_couples_indexed = []

for indices_tuple in similarity_couples:
    valeurs_tuple = [spike_id_list[idx] for idx in indices_tuple]
    
    if valeurs_tuple[0] in selected_spike_id_list and valeurs_tuple[1] in selected_spike_id_list :
        similarity_couples_indexed.append(valeurs_tuple)

print(rf"Similarity couples to check manually {similarity_couples_indexed}")

print("Next step = manually curate with phy")
print("Select what units to merge and write them in a list of lists")

for couple in similarity_couples_indexed:
    plot_maker(sorter_result,we,couple)

#%% AFTER MANUAL CURATION
"""
- Next step = manually curate with phy 
- Select what units to merge and write them in a list of lists
"""


# If you want to plot units locations

# for units in similarity_couples_indexed:
#     sw.plot_unit_locations(we,unit_ids=[units[0], units[1]])

#If you want to plot waveforms template

# for unit in selected_spike_id_list:
#     template = we.get_template(unit_id=unit, mode='median')
#     plt.figure()
#     plt.plot(template)
#     plt.title(rf"Unit # {unit}")
    
    
    

units_to_merge = [[48, 57]]

units_to_remove = [39, 40]

# definitive curated units list
if len(units_to_merge) > 0:
    clean_sorting = MergeUnitsSorting(sorter_result,units_to_merge).remove_units(spikes_not_passing_quality_metrics).remove_units(units_to_remove)
else:
    clean_sorting = sorter_result.remove_units(spikes_not_passing_quality_metrics).remove_units(units_to_remove)

# save the final curated spikesorting results
save_path = rf"{sorter_folder}\curated"
clean_sorting_saved = clean_sorting.save_to_folder(save_path)

#Get waveform from signal
clean_we = si.extract_waveforms(signal, clean_sorting,folder=rf"{sorter_folder}\curated\waveforms",load_if_exists=True)






#save units_to_merge and units_toremove (and units pre-fitlered by metrics)
curation_infos = {
    'units_merged' : units_to_merge,
    'units_removed' : units_to_remove,
    'similarity' : similarity_couples_indexed,
    'not_passing_quality_metrics' : spikes_not_passing_quality_metrics
    
    }

pickle.dump(curation_infos, open(rf"{sorter_folder}\curated\curated_infos.pickle", "wb"))

#%% export to phy
#export to phy

# save_folder_phy = rf"{sorter_folder}\curated\phy"
# sexp.export_to_phy(clean_we, output_folder=save_folder_phy, remove_if_exists=True)


#%% waveform plots
unit_list = clean_sorting.get_unit_ids()

count=0
for i in fractionner_liste(unit_list,5):
    count = count +1
    sw.plot_unit_templates(clean_we, unit_ids=i)
    
    savepath = rf"{sorter_folder}\curated\processing_data\waveforms\plots\multiple_unit_{count}.svg"
    Check_Save_Dir(os.path.dirname((savepath)))
    plt.savefig(savepath)

"""
Individual templates
"""
for i in unit_list:
    sw.plot_unit_templates(clean_we, unit_ids=np.array([i]))
    
    savepath = rf"{sorter_folder}\curated\processing_data\waveforms\plots\unit_{i}.png"
    Check_Save_Dir(os.path.dirname((savepath)))
    plt.savefig(savepath)



"""
Units location

0,0 = site 7 (SI)
tip = -75µm

So depth 0 = stereotaxy_depth - 75
0 = 825 µm

"""
units_location = spost.compute_unit_locations(clean_we)
plt.figure()
plt.scatter(units_location[:,0], 825-units_location[:,1])

for i, name in enumerate(unit_list):
    plt.text(units_location[i,0], 825-units_location[i,1], name, fontsize=15, ha='center', va='bottom')

plt.gca().invert_yaxis()
plt.title("Unit position")


savepath = rf"{sorter_folder}\curated\processing_data\waveforms\plots\Unit_locations.svg"
Check_Save_Dir(os.path.dirname((savepath)))
plt.savefig(savepath)

# spike_location = spost.compute_spike_locations(we)

"""
waveform parameters plot
"""

template_metrics = spost.compute_template_metrics(clean_we)
plt.figure()
sns.scatterplot(template_metrics, x='peak_to_valley',y='half_width')

for unit in unit_list:
    plt.annotate(unit, (template_metrics['peak_to_valley'][unit], template_metrics['half_width'][unit]))


savepath = rf"{sorter_folder}\curated\processing_data\waveforms\plots\waveforms_parameters.svg"
Check_Save_Dir(os.path.dirname((savepath)))
plt.savefig(savepath)

"""
Correlograms
"""

#TODO : correlogramms
"""
corr = spost.compute_correlograms(clean_sorting)

correlogram = corr[0][unit_list[4], unit_list[1], :]

# Accédez aux bords des bins
bin_edges = corr[1]

# Tracez l'histogramme
plt.figure(figsize=(8, 4))
plt.bar(bin_edges[:-1], correlogram, width=bin_edges[1] - bin_edges[0], align='center')
plt.xlabel('Temps (ms)')
plt.ylabel('Fréquence')
plt.title('Correlogramme entre l\'unité  et l\'unité ')
plt.grid(True)
plt.show()
"""

#%% Export datas in dataframes
"""
units location
template metrics
pca?

other df
export waveforms template
"""
units_location = spost.compute_unit_locations(clean_we)
template_metrics = spost.compute_template_metrics(clean_we)
unit_list = clean_sorting.get_unit_ids()

# pca_components = spost.compute_principal_components(we)

df = pd.DataFrame({
    "Unit" : unit_list,
    "Unit position x" : units_location[:,0],
    "Unit depth" : units_location[:,1],
    "peak_to_valley" : template_metrics['peak_to_valley'],
    "peak_trough_ratio" : template_metrics['peak_trough_ratio'],
    "half_width" : template_metrics['half_width'],
    "repolarization_slope" : template_metrics['repolarization_slope'],
    'recovery_slope' : template_metrics['recovery_slope'],
    
    }
    
    )

savepath = rf"{sorter_folder}\curated\processing_data\units_metrics.xlsx"
Check_Save_Dir(os.path.dirname((savepath)))
df.to_excel(savepath)

for unit in unit_list:
    templates = clean_we.get_all_templates(unit_ids=[unit])[0]
    df_templates = pd.DataFrame(templates)
    
    savepath = rf"{sorter_folder}\curated\processing_data\waveforms\Unit_{unit}_wf.xlsx"
    Check_Save_Dir(os.path.dirname((savepath)))
    
    df_templates.to_excel(savepath)
