# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 11:16:27 2023

@author: pierre.LE-CABEC
"""

import pandas as pd
import os
import spikeinterface as si  # import core only
import spikeinterface.sorters as ss
import spikeinterface.widgets as sw
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from neo.core import SpikeTrain
from viziphant.statistics import plot_time_histogram
from viziphant.rasterplot import rasterplot_rates
from elephant.statistics import time_histogram
from matplotlib.widgets import Button
import numpy as np
import quantities as pq
import math
import warnings
import pickle
import itertools

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


def PCA_max_amp(waveform_path, spike_path, mouse, delay, nb_of_channel_for_cor=4, corr_resutl_thr=0.7, share_spike_thr=5, plot_distribution=False):
    if delay == 'Fixed_Delay':
        delay = 'Fixed Delay'
    elif delay == 'Random_Delay':
        delay = 'Random Delay'
    else:
        raise ValueError(f'Unrecognize Delay({delay})')
        
    waveform_path = f'{waveform_path}\{delay}'
    spike_path = f'{spike_path}\{delay}'
    file_list = [(f'{waveform_path}\{file}', f'{spike_path}\{file}') for file in os.listdir(waveform_path) if file.split('.')[-1] == 'xlsx' and file.split('_')[0] == str(mouse) and file.split('_')[1] == delay]
    
    
    max_per_channel_dict = {
                       'channel 0': [],
                       'channel 1': [],
                       'channel 2': [],
                       'channel 3': [],
                       'channel 4': [],
                       'channel 5': [],
                       'channel 6': [],
                       'channel 7': [],
                       'channel 8': [],
                       'channel 9': [],
                       'channel 10': [],
                       'channel 11': [],
                       'channel 12': [],
                       'channel 13': [],
                       'channel 14': [],
                       'channel 15': [],
                       }
    
    
    waveform_spike_list = []
    for waveform_file, spike_file in file_list:
        
        waveform_base_name = int(os.path.basename(waveform_file).strip('.xlsx').split('Unit_')[-1].split(' ')[0])
        spike_base_name = int(os.path.basename(spike_file).strip('.xlsx').split('Unit_')[-1].split(' ')[0])
        if waveform_base_name != spike_base_name:
            raise ValueError(f'waveform_base_name({waveform_base_name}) does not match spike_base_name({spike_base_name})')
        else:
            unit_base_name = waveform_base_name
        
        waveform = pd.read_excel(waveform_file)
        waveform_higher_chan = waveform
        # waveform_higher_chan = pd.DataFrame()
        # while len(waveform.columns) > (16-nb_of_channel_for_cor):
        #     higher_channel = waveform.min().idxmin()
        #     waveform_higher_chan[higher_channel] = waveform[higher_channel]
        #     waveform = waveform.drop(higher_channel, axis=1)
        
        # for channel in range(16):
        #     if channel not in waveform_higher_chan.columns:
        #         waveform_higher_chan[channel] = 0
                
        # waveform_higher_chan = waveform_higher_chan.sort_index(axis=1)

        spike_time = pd.read_excel(spike_file)
        waveform_spike_list.append((unit_base_name, np.array(waveform_higher_chan), spike_time))
        
        max_per_channel = waveform.abs().max()
        for channel, value in enumerate(max_per_channel):
            max_per_channel_dict[f'channel {channel}'].append(value)
    
    supsect_combination_list = []
    for combination in itertools.combinations(waveform_spike_list, 2):
        unit_name1 = combination[0][0]
        flatten_waveform_array1 = combination[0][1].flatten()
        df_unit_1 = combination[0][2]
        
        unit_name2 = combination[1][0]
        flatten_waveform_array2 = combination[1][1].flatten()
        df_unit_2 = combination[1][2]
        
        corr_resutl = np.corrcoef(flatten_waveform_array1, flatten_waveform_array2)[0, 1] 
        
        smaller_len = len(df_unit_1) if len(df_unit_1) < len(df_unit_2) else len(df_unit_2)
        joined_df = pd.merge(df_unit_1, df_unit_2, on=['concatenated_time'])
        share_spike_proportion = (len(joined_df)/smaller_len)*100
        
        supsect_combination_list.append((unit_name1, unit_name2, round(corr_resutl, 3), round(share_spike_proportion, 1)))

   
    supsect_combination_list = sorted(supsect_combination_list, key=lambda x: x[2], reverse=True)
    combination_name = [f'{name1}-{name2}' for name1, name2,_,_ in supsect_combination_list]
    coef_cor_list = [coef_cor for _, _, coef_cor, _ in supsect_combination_list]
    share_spike_list = [share_spike for _, _, _, share_spike in supsect_combination_list]
    
    color_list = []
    for unit_name1, unit_name2, corr_resutl, share_spike_proportion in supsect_combination_list:
        if corr_resutl > corr_resutl_thr and share_spike_proportion > share_spike_thr:
            print(f'Matching unit detected: {unit_name1}-{unit_name2}, correlation: {corr_resutl}, Pourcentage of share spike: {share_spike_proportion}%')
            color_list.append('r')
        else:
           color_list.append('b') 
           
    if plot_distribution:
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].bar(combination_name, share_spike_list, color=color_list)
        ax[0].set_title('Pourcentage of shared spike (%)')
        ax[0].axhline(share_spike_thr, color='r', alpha=0.5, linestyle='--')
        ax[1].set_title('Coefficient of waveform correlation')
        ax[1].bar(combination_name, coef_cor_list, color=color_list)
        ax[1].axhline(corr_resutl_thr, color='r', alpha=0.5, linestyle='--')
        plt.tight_layout()
        
def frozenset_dict_maker(higher_channel_dict):
    frozenset_dict = {}

    # Group identical lists and associate with original IDs
    for k, v in higher_channel_dict.items():
        key = tuple(v)
        if key in frozenset_dict:
            frozenset_dict[key].append(k)
        else:
            frozenset_dict[key] = [k]
    
    return frozenset_dict

def move_and_arrange_figures(nb_fig_layer=8):
    """
    Move and arrange all open figures on the screen.
    """
    # Get screen size
    manager = plt.get_current_fig_manager()
    screen = manager.window.screen()
    screen_width = screen.size().width()
    screen_height = screen.size().height()

    # Get all open figures
    figures = [plt.figure(num) for num in plt.get_fignums()]
    nb_fig_display = 0
    start_ind = 0    
    end_ind = nb_fig_layer
    while nb_fig_display < len(figures):
        current_figures = figures[start_ind: end_ind]
        nb_fig_display += len(current_figures)
        start_ind = end_ind 
        end_ind = end_ind + nb_fig_layer
        
        # Compute grid size
        # if len(figures) < max_fig:
        #     num_figs = len(current_figures)
        # else:
        #     num_figs = max_fig
            
        # ncols = math.ceil(math.sqrt(num_figs))
        ncols = math.ceil(nb_fig_layer/2)
        nrows = math.ceil(nb_fig_layer / ncols)
    
        # Compute figure size
        fig_width = screen_width // ncols
        fig_height = screen_height // nrows
    
        for i, fig in enumerate(current_figures):
            # Compute position (note: (0, 0) is the top left corner)
            fig_x = (i % ncols) * fig_width
            fig_y = (i // ncols) * fig_height
    
            # Set figure position
            fig.canvas.manager.window.move(fig_x, fig_y)
    
            # Set figure size
            fig.set_size_inches((fig_width/100), (fig_height/96)-0.8)
            fig.tight_layout()

def plot_maker(sorter, we, save, unit_id, sorter_name, mouse, delay):
    fig = plt.figure(figsize=(25, 13))
    gs = GridSpec(nrows=3, ncols=6)
    fig.suptitle(f'{sorter_name}\n{mouse} {delay}\nunits {unit_id} (Total spike {sorter.get_total_num_spikes()[unit_id]})',)
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
            current_spike_train_list.append(SpikeTrain(current_spike_train[indices]*pq.s, t_stop=9))
            # Remove the appended elements from the array
            current_spike_train = np.delete(current_spike_train, indices)
            # Subtract 9 from all remaining elements
        current_spike_train -= 9
    bin_size = 100
    histogram = time_histogram(current_spike_train_list, bin_size=bin_size*pq.ms, output='mean')
    histogram = histogram*(1000/bin_size)
    ax1.axvspan(0, 0.5, color='green', alpha=0.3)
    ax1.axvspan(1.5, 2, color='green', alpha=0.3)
    ax6.axvspan(0, 0.5, color='green', alpha=0.3)
    ax6.axvspan(1.5, 2, color='green', alpha=0.3)

    if delay == 'Fixed_Delay':
        ax1.axvspan(2.5, 2.65, color='red', alpha=0.3)
        ax6.axvspan(2.5, 2.65, color='red', alpha=0.3)
    elif delay == 'Random_Delay':
        ax1.axvspan(2.4, 2.55, color='red', alpha=0.3)
        ax1.axvspan(2.9, 3.05, color='red', alpha=0.3)
        ax6.axvspan(2.4, 2.55, color='red', alpha=0.3)
        ax6.axvspan(2.9, 3.05, color='red', alpha=0.3)

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
        
    return fig

def button_callback(sorter_info_dict, mouse, delay, saving_spike_path, saving_waveform_path, saving_summary_plot, unit_base_nb):
    plt.close('all')  # Close all figure
    if sorter_info_dict['sorter_name'] != 'pass':
        if delay == 'Fixed_Delay':
            recording_delay = 'Fixed Delay'
        elif delay == 'Random_Delay':
            recording_delay = 'Random Delay'
        else:
            raise ValueError(f'Unrecognize Delay({delay})')
        unit_id = sorter_info_dict['unit_id']
        sorter_info_dict['fig'].set_figheight(13)
        sorter_info_dict['fig'].set_figwidth(25)
        sorter_info_dict['fig'].savefig(fr"{saving_summary_plot}\{recording_delay}\{mouse}_{recording_delay}_Unit_{unit_base_nb} (Org_id_{unit_id}_{sorter_info_dict['sorter_name']}).pdf")
        
        wafevorme_df = pd.DataFrame(sorter_info_dict['we'].get_template(unit_id))
        wafevorme_df.to_excel(fr"{saving_waveform_path}\{recording_delay}\{mouse}_{recording_delay}_Unit_{unit_base_nb} (Org_id_{unit_id}_{sorter_info_dict['sorter_name']}).xlsx", index=False)
        
        recording_path = f'C:\local_data\Paper\Data\concaneted_recording\{mouse}_{recording_delay}'
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
        
        current_unit_spike_time_summary_df_list.to_excel(fr"{saving_spike_path}\{recording_delay}\{mouse}_{recording_delay}_Unit_{unit_base_nb} (Org_id_{unit_id}_{sorter_info_dict['sorter_name']}).xlsx")
        
def unit_high_channel_ploting(higher_chanel_order_dict, sorter_dict, mouse, delay, saving_spike_path, saving_waveform_path, saving_summary_plot, unit_base_nb):
    
    for higher_chanel_order, unit_list in higher_chanel_order_dict.items():
        sorter_info_list = []
        for unit in unit_list:
            sorter_name = unit.split('_')[0]
            unit_id = int(unit.split('_')[1])
            sorter = sorter_dict[sorter_name]['sorter']
            we = sorter_dict[sorter_name]['we']
            fig = plot_maker(sorter, we, False, unit_id, sorter_name, mouse, delay)
            sorter_info_list.append({'sorter': sorter, 'we': we, 'sorter_name': sorter_name, 'unit_id': unit_id, 'fig': fig, 'highest_channel': higher_chanel_order[0]})
                    
        if len(sorter_info_list):
            move_and_arrange_figures(nb_fig_layer=len(unit_list))
            fig, axes = plt.subplots(1, len(sorter_info_list)+1, figsize=(6, 2))
            if len(sorter_info_list) > 0:
                button1 = Button(axes[0], f"{sorter_info_list[0]['sorter_name']}_{sorter_info_list[0]['unit_id']}")
                button1.on_clicked(lambda event: button_callback(sorter_info_list[0], mouse, delay, saving_spike_path, saving_waveform_path, saving_summary_plot, unit_base_nb))
            if len(sorter_info_list) > 1:
                button2 = Button(axes[1], f"{sorter_info_list[1]['sorter_name']}_{sorter_info_list[1]['unit_id']}")
                button2.on_clicked(lambda event: button_callback(sorter_info_list[1], mouse, delay, saving_spike_path, saving_waveform_path, saving_summary_plot, unit_base_nb))
            if len(sorter_info_list) > 2:
                button3 = Button(axes[2], f"{sorter_info_list[2]['sorter_name']}_{sorter_info_list[2]['unit_id']}")
                button3.on_clicked(lambda event: button_callback(sorter_info_list[2], mouse, delay, saving_spike_path, saving_waveform_path, saving_summary_plot, unit_base_nb))
            if len(sorter_info_list) > 3:
                button4 = Button(axes[3], f"{sorter_info_list[3]['sorter_name']}_{sorter_info_list[3]['unit_id']}")
                button4.on_clicked(lambda event: button_callback(sorter_info_list[3], mouse, delay, saving_spike_path, saving_waveform_path, saving_summary_plot, unit_base_nb))
            if len(sorter_info_list) > 4:
                button5 = Button(axes[4], f"{sorter_info_list[4]['sorter_name']}_{sorter_info_list[4]['unit_id']}")
                button5.on_clicked(lambda event: button_callback(sorter_info_list[4], mouse, delay, saving_spike_path, saving_waveform_path, saving_summary_plot, unit_base_nb))
            if len(sorter_info_list) > 5:
                button6 = Button(axes[5], f"{sorter_info_list[5]['sorter_name']}_{sorter_info_list[5]['unit_id']}")
                button6.on_clicked(lambda event: button_callback(sorter_info_list[5], mouse, delay, saving_spike_path, saving_waveform_path, saving_summary_plot, unit_base_nb))
            if len(sorter_info_list) > 6:
                button7 = Button(axes[6], f"{sorter_info_list[6]['sorter_name']}_{sorter_info_list[6]['unit_id']}")
                button7.on_clicked(lambda event: button_callback(sorter_info_list[6], mouse, delay, saving_spike_path, saving_waveform_path, saving_summary_plot, unit_base_nb))
            if len(sorter_info_list) > 7:
                button8 = Button(axes[7], f"{sorter_info_list[7]['sorter_name']}_{sorter_info_list[7]['unit_id']}")
                button8.on_clicked(lambda event: button_callback(sorter_info_list[7], mouse, delay, saving_spike_path, saving_waveform_path, saving_summary_plot, unit_base_nb))
            if len(sorter_info_list) > 8:
                button9 = Button(axes[8], f"{sorter_info_list[8]['sorter_name']}_{sorter_info_list[8]['unit_id']}")
                button9.on_clicked(lambda event: button_callback(sorter_info_list[8], mouse, delay, saving_spike_path, saving_waveform_path, saving_summary_plot, unit_base_nb))
            if len(sorter_info_list) > 9:
                button10 = Button(axes[9], f"{sorter_info_list[9]['sorter_name']}_{sorter_info_list[9]['unit_id']}")
                button10.on_clicked(lambda event: button_callback(sorter_info_list[9], mouse, delay, saving_spike_path, saving_waveform_path, saving_summary_plot, unit_base_nb))
            if len(sorter_info_list) > 10:
                button11 = Button(axes[10], f"{sorter_info_list[10]['sorter_name']}_{sorter_info_list[10]['unit_id']}")
                button11.on_clicked(lambda event: button_callback(sorter_info_list[10], mouse, delay, saving_spike_path, saving_waveform_path, saving_summary_plot, unit_base_nb))
            if len(sorter_info_list) > 11:
                button12 = Button(axes[11], f"{sorter_info_list[11]['sorter_name']}_{sorter_info_list[11]['unit_id']}")
                button12.on_clicked(lambda event: button_callback(sorter_info_list[11], mouse, delay, saving_spike_path, saving_waveform_path, saving_summary_plot, unit_base_nb))
            if len(sorter_info_list) > 12:
                button13 = Button(axes[12], f"{sorter_info_list[12]['sorter_name']}_{sorter_info_list[12]['unit_id']}")
                button13.on_clicked(lambda event: button_callback(sorter_info_list[12], mouse, delay, saving_spike_path, saving_waveform_path, saving_summary_plot, unit_base_nb))
            
            buttonPass = Button(axes[-1], "Pass")
            buttonPass.on_clicked(lambda event: button_callback({'sorter_name': 'pass'}, mouse, delay, saving_spike_path, saving_waveform_path, saving_summary_plot, unit_base_nb))

            plt.show()
            if not plt.waitforbuttonpress():
                plt.pause(0.5) 
                unit_base_nb += 1
    
    return unit_base_nb

def higher_chanel_order_dict(sorter_list, saving_spike_path, saving_waveform_path, saving_summary_plot, mouse, delay, unit_base_nb=0, number_of_channel_for_comp=2):

    higher_channel_dict = {}
    sorter_dict = {}
    if delay == 'Fixed_Delay':
        recording_delay = 'Fixed Delay'
    elif delay == 'Random_Delay':
        recording_delay = 'Random Delay'
    else:
        raise ValueError(f'Unrecognize Delay({delay})')
    recording_path = fr'C:\local_data\Paper\Data\concaneted_recording\{mouse}_{recording_delay}'
    os.chdir(recording_path)

    multirecording = si.BinaryFolderRecording(f'{recording_path}/concatenated_recording')
    multirecording.annotate(is_filtered=True)
    
    we_list = []
    we_name_list = []
    for sorter in sorter_list:
        print(sorter)
        if sorter == 'comp_mult_2_tridesclous_herdingspikes_mountainsort4':
            sorter_mini = 'comp'
        elif sorter == 'herdingspikes':
            sorter_mini = 'her'
        elif sorter == 'mountainsort4':
            sorter_mini = 'moun'
        elif sorter == 'tridesclous':
            sorter_mini = 'tdc'
        else:
            sorter_mini = sorter

        spike_folder = f'C:\local_data\Paper\Data\spike\{mouse}_{delay}\{sorter}'
        if sorter_mini == 'comp':
            sorter_result = ss.NpzSortingExtractor.load_from_folder(f'{spike_folder}\sorter')
        else:
            sorter_result = ss.NpzSortingExtractor.load_from_folder(f'{spike_folder}\sorter\in_container_sorting')
     
        we = si.WaveformExtractor.load_from_folder(f'{spike_folder}\we', sorting=sorter_result)
        we_list.append(we)
        we_name_list.append(sorter_mini)
        for unit in sorter_result.get_unit_ids():
            waveform = pd.DataFrame(we.get_template(unit)) 
            waveform = waveform.abs()

            higher_channel_list = []
            while len(waveform.columns) > (16-number_of_channel_for_comp):
                higher_channel = waveform.max().idxmax()
                waveform = waveform.drop(higher_channel, axis=1)
                higher_channel_list.append(int(higher_channel))
           
            higher_channel_dict[f'{sorter_mini}_{unit}'] = higher_channel_list
            if sorter_mini not in sorter_dict.keys():
                sorter_dict[sorter_mini] = {'sorter': sorter_result,
                                            'we': we}            

    higher_chanel_order_dict = frozenset_dict_maker(higher_channel_dict)
    unit_base_nb = unit_high_channel_ploting(higher_chanel_order_dict, sorter_dict, mouse, delay, saving_spike_path, saving_waveform_path, saving_summary_plot, unit_base_nb)



number_of_channel_for_comp = 2
#'173', '174', '176', '6401', '6402', '6409', '6924', '6928', '6456', '6457'
mouse = '173'
#'Fixed_Delay', 'Random_Delay'
delay = 'Random_Delay'
unit_base_nb = 0
sorter_list = ['comp_mult_2_tridesclous_herdingspikes_mountainsort4', 'herdingspikes', 'mountainsort4', 'tridesclous']
saving_spike_path = rf'C:\local_data\Paper\Data\spike\spike_time'
saving_waveform_path = rf'C:\local_data\Paper\Data\spike\waveform'
saving_summary_plot = rf'C:\local_data\Paper\Data\spike\unit_summary_plot'

# higher_chanel_order_dict(sorter_list, saving_spike_path, saving_waveform_path, saving_summary_plot, mouse, delay, unit_base_nb, number_of_channel_for_comp)
PCA_max_amp(saving_waveform_path, saving_spike_path, mouse, delay, corr_resutl_thr=0.8, share_spike_thr=5, plot_distribution=True)
