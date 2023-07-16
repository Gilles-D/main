# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 18:14:52 2023

@author: Gil
"""
import spikeinterface as si
import spikeinterface.extractors as se 
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.comparison as sc
import spikeinterface.exporters as sexp
import spikeinterface.widgets as sw

import os

import probeinterface as pi
from probeinterface.plotting import plot_probe

import warnings
warnings.simplefilter("ignore")


import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from matplotlib.widgets import Button

from viziphant.statistics import plot_time_histogram
from viziphant.rasterplot import rasterplot_rates
from elephant.statistics import time_histogram
from neo.core import SpikeTrain
from quantities import s, ms

import pandas as pd
import math





def higher_channel_order_dict(recording_path,spikesorting_results_path, sorter_list, saving_spike_path, saving_waveform_path, saving_summary_plot, unit_base_nb=0, number_of_channel_for_comp=2):
    # Initialize dictionaries
    higher_channel_dict = {}
    sorter_dict = {}

    # Construct recording path based on 'mouse' and 'recording_delay'
    recording_path = recording_path

    # Change current working directory to recording_path
    os.chdir(recording_path)

    # Load binary recording 'multirecording' from the 'concatenated_recording' folder
    multirecording = si.BinaryFolderRecording(recording_path)
    multirecording.annotate(is_filtered=True)

    # Initialize waveform extractor lists
    we_list = []
    we_name_list = []
    
    #check saving directories
    Check_Save_Dir(saving_spike_path)
    Check_Save_Dir(saving_waveform_path)
    Check_Save_Dir(saving_summary_plot)

    # Loop over each sorter in sorter_list
    for sorter in sorter_list:
        print(sorter)
        # Assign abbreviated names to sorters
        if sorter == 'comp_mult_2_tridesclous_spykingcircus_mountainsort4':
            sorter_mini = 'comp'
        elif sorter == 'herdingspikes':
            sorter_mini = 'her'
        elif sorter == 'mountainsort4':
            sorter_mini = 'moun'
        elif sorter == 'tridesclous':
            sorter_mini = 'tdc'
        else:
            sorter_mini = sorter

        # Construct spike folder path based on 'mouse', 'delay', and 'sorter'
        spike_folder = rf'{spikesorting_results_path}/{sorter}'
        
        if sorter_mini == 'comp':
            # Load sorting results for 'comp' sorter
            print(f'{spike_folder}\sorter')
            sorter_result = ss.NpzSortingExtractor.load_from_folder(f'{spike_folder}\sorter')
        else:
            # Load sorting results for other sorters
            print(f'{spike_folder}\in_container_sorting')
            sorter_result = ss.NpzSortingExtractor.load_from_folder(f'{spike_folder}\in_container_sorting')

        # Load waveform extractor ('we') from the 'we' folder
        we = si.WaveformExtractor.load_from_folder(f'{spike_folder}\we', sorting=sorter_result)

        # Add waveform extractor and sorter name to the respective lists
        we_list.append(we)
        we_name_list.append(sorter_mini)

        # Loop over each unit in the sorter results
        for unit in sorter_result.get_unit_ids():
            waveform = pd.DataFrame(we.get_template(unit))
            waveform = waveform.abs()

            higher_channel_list = []

            # Extract channels with highest amplitudes until the desired number is reached
            while len(waveform.columns) > (16 - number_of_channel_for_comp):
                higher_channel = waveform.max().idxmax()
                waveform = waveform.drop(higher_channel, axis=1)
                higher_channel_list.append(int(higher_channel))

            # Store higher channel list in higher_channel_dict
            higher_channel_dict[f'{sorter_mini}_{unit}'] = higher_channel_list

            # Store sorter result and waveform extractor in sorter_dict
            if sorter_mini not in sorter_dict.keys():
                sorter_dict[sorter_mini] = {'sorter': sorter_result, 'we': we}

    # Call frozenset_dict_maker to create an immutable dictionary from higher_channel_dict
    higher_channel_order_dict = frozenset_dict_maker(higher_channel_dict)

    # Call unit_high_channel_ploting with appropriate arguments
    unit_base_nb = unit_high_channel_ploting(higher_channel_order_dict, sorter_dict, saving_spike_path, saving_waveform_path, saving_summary_plot, unit_base_nb)



def frozenset_dict_maker(higher_channel_dict):
    frozenset_dict = {}
    # Groupe les listes identiques et les associe aux ID d'origine
    for k, v in higher_channel_dict.items():
        key = tuple(v)
        if key in frozenset_dict:
            frozenset_dict[key].append(k)
        else:
            frozenset_dict[key] = [k]
    
    return frozenset_dict


def plot_maker(sorter, we, save, unit_id, sorter_name):
    # Création de la figure et de la grille pour les sous-graphiques
    fig = plt.figure(figsize=(25, 13))
    gs = GridSpec(nrows=3, ncols=6)
    
    # Titre principal de la figure avec les informations sur le trieur, la souris, le délai et l'unité
    fig.suptitle(f'{sorter_name}\nunits {unit_id} (Total spike {sorter.get_total_num_spikes()[unit_id]})',)
    
    # Création des sous-graphiques
    ax0 = fig.add_subplot(gs[0, 0:3])
    ax1 = fig.add_subplot(gs[0, 3:7])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1], sharey=ax3)
    ax5 = fig.add_subplot(gs[2, 2], sharey=ax3)
    ax6 = fig.add_subplot(gs[2, 3:6])
    
    # Configuration des titres des sous-graphiques
    ax1.set_title('Mean firing rate during a trial')
    ax2.set_title('Waveform of the unit')
    
    # Tracé de l'autocorrélogramme avec plot_autocorrelograms de spikeextractors
    sw.plot_autocorrelograms(sorter, unit_ids=[unit_id], axes=ax0, bin_ms=1, window_ms=200)
    ax0.set_title('Autocorrelogram')
    
    # Traitement du spike train pour le découper en essais de durée correcte en secondes
    current_spike_train = sorter.get_unit_spike_train(unit_id) / sorter.get_sampling_frequency()
    current_spike_train_list = []
    while len(current_spike_train) > 0:
        indices = np.where(current_spike_train < 9)[0]  # Indices des éléments inférieurs à 9 (durée de l'essai)
        if len(indices) > 0:
            current_spike_train_list.append(SpikeTrain(current_spike_train[indices] * s, t_stop=9))
            current_spike_train = np.delete(current_spike_train, indices)
        current_spike_train -= 9
    
    bin_size = 100
    histogram = time_histogram(current_spike_train_list, bin_size=bin_size * ms, output='mean')
    histogram = histogram * (1000 / bin_size)
    
    # Configuration des zones de couleur sur les sous-graphiques ax1 et ax6 en fonction du délai
    ax1.axvspan(0, 0.5, color='green', alpha=0.3)
    ax1.axvspan(1.5, 2, color='green', alpha=0.3)
    ax6.axvspan(0, 0.5, color='green', alpha=0.3)
    ax6.axvspan(1.5, 2, color='green', alpha=0.3)
    
    # Tracé de l'histogramme temporel
    plot_time_histogram(histogram, units='s', axes=ax1)
    
    # Tracé de la densité des formes d'ondes du trieur
    sw.plot_unit_waveforms_density_map(we, unit_ids=[unit_id], ax=ax2)
    
    # Extraction du modèle de la forme d'onde de l'unité
    template = we.get_template(unit_id=unit_id).copy()
    for current_ax in [ax3, ax4, ax5]:
        max_channel = np.argmax(np.abs(template)) % template.shape[1]
        template[:, max_channel] = 0
        mean_residual = np.mean(np.abs((we.get_waveforms(unit_id=unit_id)[:, :, max_channel] - we.get_template(unit_id=unit_id)[:, max_channel])), axis=0)
        current_ax.plot(mean_residual)
        current_ax.plot(we.get_template(unit_id=unit_id)[:, max_channel])
        current_ax.set_title('Mean residual of the waveform for channel ' + str(max_channel))
    
    plt.tight_layout()
    
    # Tracé du rasterplot des taux de décharge
    rasterplot_rates(current_spike_train_list, ax=ax6, histscale=0.1) 
    
    return fig


def button_callback(sorter_info_dict, saving_spike_path, saving_waveform_path, saving_summary_plot, unit_base_nb):
    plt.close('all')  # Close all figure
    if sorter_info_dict['sorter_name'] != 'pass':

        unit_id = sorter_info_dict['unit_id']
        sorter_info_dict['fig'].set_figheight(13)
        sorter_info_dict['fig'].set_figwidth(25)
        sorter_info_dict['fig'].savefig(fr"{saving_summary_plot}\Unit_{unit_base_nb} (Org_id_{unit_id}_{sorter_info_dict['sorter_name']}).pdf")
        
        wafevorme_df = pd.DataFrame(sorter_info_dict['we'].get_template(unit_id))
        wafevorme_df.to_excel(fr"{saving_waveform_path}\Unit_{unit_base_nb} (Org_id_{unit_id}_{sorter_info_dict['sorter_name']}).xlsx", index=False)
        
        recording_path = rf'{concatenated_signals_path}/{recording_name}'
        
        print(fr'{recording_path}/concatenated_recording_trial_time_index_df.pickle')
        
        with open(fr'{recording_path}/concatenated_recording_trial_time_index_df.pickle', 'rb') as handle:
            
            trial_time_index_df = pickle.load(handle)
            
        concatenated_spike_times = sorter_info_dict['sorter'].get_unit_spike_train(unit_id)
        real_spike_time = trial_time_index_df[trial_time_index_df.index.isin(concatenated_spike_times)]
        if len(concatenated_spike_times) != len(real_spike_time):
            u, c = np.unique(concatenated_spike_times, return_counts=True)
            dup = u[c > 1]
            if (len(concatenated_spike_times) - len(dup)) == len(real_spike_time):
                print(f'{len(dup)} duplicate found in unit {unit_base_nb} (orig_id {unit_id}), the duplicates has been remove from the spiketrain')
            else:
                print(f'concatenated_spike_times ({len(concatenated_spike_times)}) is not equal to real_spike_time ({len(real_spike_time)})\nand is not explain by duplicate spike ({len(dup)})')
                # raise ValueError(f'concatenated_spike_times ({len(concatenated_spike_times)}) is not equal to real_spike_time ({len(real_spike_time)})\nand is not explain by duplicate spike ({len(dup)})')
        current_unit_spike_time_summary_df_list = pd.DataFrame(real_spike_time)
        
        current_unit_spike_time_summary_df_list.to_excel(fr"{saving_spike_path}\Unit_{unit_base_nb} (Org_id_{unit_id}_{sorter_info_dict['sorter_name']}).xlsx")
        



def unit_high_channel_ploting(higher_chanel_order_dict, sorter_dict, saving_spike_path, saving_waveform_path, saving_summary_plot, unit_base_nb):
    # Parcours des combinaisons de canaux supérieurs et listes d'unités dans higher_chanel_order_dict
    for higher_chanel_order, unit_list in higher_chanel_order_dict.items():
        sorter_info_list = []
        # Parcours des unités dans la liste d'unités
        for unit in unit_list:
            # Extraction des informations de trieur et d'unité à partir de l'ID d'unité
            sorter_name = unit.split('_')[0]
            unit_id = int(unit.split('_')[1])
            sorter = sorter_dict[sorter_name]['sorter']
            we = sorter_dict[sorter_name]['we']
            
            # Création d'une figure à partir des informations de trieur et d'unité
            fig = plot_maker(sorter, we, False, unit_id, sorter_name)
            
            # Ajout des informations du trieur et de l'unité à la liste sorter_info_list
            sorter_info_list.append({'sorter': sorter, 'we': we, 'sorter_name': sorter_name, 'unit_id': unit_id, 'fig': fig, 'highest_channel': higher_chanel_order[0]})
                    
        if len(sorter_info_list):
            # Appel à la fonction move_and_arrange_figures pour déplacer et organiser les figures
            move_and_arrange_figures(nb_fig_layer=len(unit_list))
            
            # Création d'une figure et de boutons pour chaque trieur et unité dans sorter_info_list
            fig, axes = plt.subplots(1, len(sorter_info_list)+1, figsize=(6, 2))
            for i, sorter_info in enumerate(sorter_info_list):
                button = Button(axes[i], f"{sorter_info['sorter_name']}_{sorter_info['unit_id']}")
                button.on_clicked(lambda event: button_callback(sorter_info,saving_spike_path, saving_waveform_path, saving_summary_plot, unit_base_nb))
                
            # Création d'un bouton "Pass" pour ignorer les combinaisons
            buttonPass = Button(axes[-1], "Pass")
            buttonPass.on_clicked(lambda event: button_callback({'sorter_name': 'pass'},saving_spike_path, saving_waveform_path, saving_summary_plot, unit_base_nb))

            plt.show()
            
            # Attente d'un appui sur un bouton ou d'une pause avant de passer à l'unité suivante
            if not plt.waitforbuttonpress():
                plt.pause(0.5) 
                unit_base_nb += 1
    
    return unit_base_nb


def move_and_arrange_figures(nb_fig_layer=8):
    """
    Déplace et organise toutes les figures ouvertes à l'écran.
    """
    # Obtenir la taille de l'écran
    manager = plt.get_current_fig_manager()
    screen = manager.window.screen()
    screen_width = screen.size().width()
    screen_height = screen.size().height()
    
    # Obtenir toutes les figures ouvertes
    figures = [plt.figure(num) for num in plt.get_fignums()]
    nb_fig_display = 0
    start_ind = 0
    end_ind = nb_fig_layer
    
    while nb_fig_display < len(figures):
        current_figures = figures[start_ind: end_ind]
        nb_fig_display += len(current_figures)
        start_ind = end_ind
        end_ind = end_ind + nb_fig_layer
        
        # Calcul de la taille de la grille
        ncols = math.ceil(nb_fig_layer / 2)
        nrows = math.ceil(nb_fig_layer / ncols)
    
        # Calcul de la taille de la figure
        fig_width = screen_width // ncols
        fig_height = screen_height // nrows
    
        for i, fig in enumerate(current_figures):
            # Calcul de la position (note : (0, 0) est le coin supérieur gauche)
            fig_x = (i % ncols) * fig_width
            fig_y = (i // ncols) * fig_height
    
            # Définir la position de la figure
            fig.canvas.manager.window.move(fig_x, fig_y)
    
            # Définir la taille de la figure
            fig.set_size_inches((fig_width / 100), (fig_height / 96) - 0.8)
            fig.tight_layout()


def Check_Save_Dir(save_path):
    """
    Check if the save folder exists
    If not : creates it
    
    """
    import os
    isExist = os.path.exists(save_path)
    if not isExist:
        os.makedirs(save_path) #Create folder for the experience if it is not already done
    return


"""
Parameters
"""

recording_name="0012_03_07_nooptotag_allchan"
sorter_list = ['comp_mult_2_tridesclous_spykingcircus_mountainsort4', 'mountainsort4','spykingcircus', 'tridesclous']



# Saving Folder path
concatenated_signals_path=r"D:/ePhy/SI_Data/concatenated_signals"
recording_path = rf'{concatenated_signals_path}/{recording_name}'
spikesorting_results_path=rf"D:/ePhy/SI_Data/spikesorting_results/{recording_name}"


saving_spike_path = rf'{spikesorting_results_path}/spikes'
saving_waveform_path = rf'{spikesorting_results_path}/waveforms'
saving_summary_plot = rf'{spikesorting_results_path}/summary_plots'


higher_channel_order_dict(recording_path,spikesorting_results_path,sorter_list,saving_spike_path,saving_waveform_path,saving_summary_plot)
