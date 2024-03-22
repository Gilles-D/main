# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:59:35 2023

@author: Gilles Delbecq

Concatenate signal from intan files for a given session
Saves concatenated signal for spikeinterface analysis in binary format (spikesorting)

Inputs = intan files (.rhd format)
Outputs = binary format (spikeinterface readable)

"""
#%% Imports
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
import sys
import time

import probeinterface as pi
from probeinterface.plotting import plot_probe

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

def list_recording_files(path):
    """
    List all recording files (.rhd) in the specified directory and its subdirectories.
    
    Parameters:
        path (str): The directory path to search for recording files.
        
    Returns:
        list: A list of paths to all recording files found.
    """
    
    import glob
    fichiers = [fichier for fichier in glob.iglob(path + '/**/*', recursive=True) if not os.path.isdir(fichier) and fichier.endswith('.rhd')]
    
    return fichiers


def TTL_detection(TTL_starts_idx, sampling_rate):
    """
    Detects start indexes and times of TTL pulses.

    Args:
        TTL_starts_idx (numpy.ndarray): A 1D numpy array containing the start indexes of TTL pulses.
        sampling_rate (float): The sampling rate of the recording.

    Returns:
        tuple: A tuple containing two arrays:
            - start_indexes (numpy.ndarray): A 1D numpy array containing the start indexes of each phase of TTL pulses.
            - start_times (numpy.ndarray): A 1D numpy array containing the start times (in seconds) of each phase of TTL pulses.
    """
    # Calculate the difference between consecutive elements
    diff_indices = np.diff(TTL_starts_idx)
    phase_indices = np.where(diff_indices != 1)[0]
    
    phase_end_indices = np.where(diff_indices != 1)[0]
    
    start_indexes = TTL_starts_idx[phase_indices] #fin de ttl
    
    start_indexes = np.insert(start_indexes, 0, TTL_starts_idx[0])
    start_times = start_indexes / sampling_rate

    return start_indexes, start_times



def trouver_changements_indices(tableau):
    """
    Détecte les index d'apparition et de fin de TTL (les momments où il passe de 0 à 1 ou de 1 à 0)
    """
    diff = np.diff(tableau.astype(int))
    return np.where(diff != 0)[0] + 1
   



def concatenate_preprocessing(recordings,saving_dir,saving_name,probe_path,excluded_sites,freq_min=300,freq_max=6000,MOCAP_200Hz_notch=True,remove_stim_artefact=True,Plotting=True):
    #Check if concatenated file already exists
    if os.path.isdir(rf'{saving_dir}/{saving_name}/'):
        print('Concatenated file already exists')
        rec_binary = si.load_extractor(rf'{saving_dir}/{saving_name}/')
    
    
    
    else:
        print('Concatenating...')
        
        
        """------------------Concatenation------------------"""
        # recordings_list=[]
        # for recording_file in recordings:
        #     recording = se.read_intan(recording_file,stream_id='0')
        #     recording.annotate(is_filtered=False)
        #     recordings_list.append(recording)
        
        
        recording_list = [se.read_intan(rhd_file, stream_id='0') for rhd_file in recordings]
        print(recording_list)
        multirecording = si.concatenate_recordings(recording_list)
        
        
        recording_info_path = os.path.dirname((os.path.dirname(recordings[0])))
        recording_info = pickle.load(open(rf'{recording_info_path}/ttl_idx.pickle', "rb"))

                 

        """------------------Set the probe------------------"""
        probe = pi.io.read_probeinterface(probe_path)
        probe = probe.probes[0]
        multirecording = multirecording.set_probe(probe)
        if Plotting==True:
            plot_probe(probe, with_device_index=True)

        
        """------------------Defective sites exclusion------------------"""
        if Plotting==True:
            sw.plot_timeseries(multirecording, channel_ids=multirecording.get_channel_ids(),time_range=[10,30])
        
        multirecording.set_channel_groups(1, excluded_sites)
        multirecording = multirecording.split_by('group')[0]
        if Plotting==True:
            sw.plot_timeseries(multirecording, channel_ids=multirecording.get_channel_ids(),time_range=[10,30])
        
        
        
                
        if remove_stim_artefact == True:
            stim_idx = recording_info['stim_ttl_on']
            multirecording = spre.remove_artifacts(multirecording,stim_idx, ms_before=1.2, ms_after=1.2,mode='linear')
            w = sw.plot_timeseries(multirecording,time_range=[stim_idx[0]/20000,(stim_idx[0]/20000)+10], segment_index=0)

            
        
        """------------------Pre Processing------------------"""
        #Bandpass filter
        recording_f = spre.bandpass_filter(multirecording, freq_min=freq_min, freq_max=freq_max)
        if Plotting==True:
            w = sw.plot_timeseries(recording_f,time_range=[10,30], segment_index=0)
        
        
        if MOCAP_200Hz_notch == True:
            for i in [num for num in range(300, 6000 + 1) if num % 200 == 0]:
                recording_f = spre.notch_filter(recording_f, freq=i)
            print("DEBUG : MOCAP Notch")
        
        
        
        #Median common ref

        recording_cmr = spre.common_reference(recording_f, reference='global', operator='median')
        print("DEBUG : CMR")

        
        if Plotting==True:
            w = sw.plot_timeseries(recording_cmr,time_range=[10,30], segment_index=0)
            
        print(rf'{saving_dir}/{saving_name}/')
        
        rec_binary = recording_cmr.save(format='binary',folder=rf'{saving_dir}/{saving_name}/', n_jobs=1, progress_bar=True, chunk_duration='1s')
        print("DEBUG : rec_binary")
       
        trial_time_index_df=pd.DataFrame({'concatenated_time':multirecording.get_times()})
        print("DEBUG : df")

        with open(rf'{saving_dir}/{saving_name}/concatenated_recording_trial_time_index_df.pickle', 'wb') as file:
            pickle.dump(trial_time_index_df, file, protocol=pickle.HIGHEST_PROTOCOL)   
            
        pickle.dump(recording_info, open(rf"{saving_dir}/{saving_name}/ttl_idx.pickle", "wb"))

    return rec_binary

    
    
def plot_maker(sorter, we, save, sorter_name, save_path,saving_name):
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
    
    for unit_id in sorter.get_unit_ids():
        fig = plt.figure(figsize=(25, 13))
        gs = GridSpec(nrows=3, ncols=6)
        fig.suptitle(f'{sorter_name}\n{saving_name}\nunits {unit_id} (Total spike {sorter.get_total_num_spikes()[unit_id]})',)
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
        if save:
            plt.savefig(fr'{save_path}\{saving_name}\{sorter_name}\we\Unit_{int(unit_id)}.pdf')
            plt.savefig(fr'{save_path}\{saving_name}\{sorter_name}\we\Unit_{int(unit_id)}.png')
            plt.close()


#%%Parameters

#####################################################################
###################### TO CHANGE ####################################
#####################################################################


#Folder containing the folders of the session
animal_id = "0035"
session_name = "0035_26_01"
saving_name=session_name

rhd_folder = rf'D:\ePhy\Intan_Data\{animal_id}\{session_name}'


#####################################################################
#Verify the following parameters and paths

probe_path=r'D:/ePhy/SI_Data/A1x16-Poly2-5mm-50s-177.json'   #INTAN Optrode
# probe_path = 'D:/ePhy/SI_Data/Buzsaki16.json'              #INTAN Buzsaki16


# Saving Folder path
saving_dir=r"D:/ePhy/SI_Data/concatenated_signals"
spikesorting_results_folder='D:\ePhy\SI_Data\spikesorting_results'


# Sites to exclude
excluded_sites = []


#%%Main script
recordings = list_recording_files(rhd_folder)

    
recording = concatenate_preprocessing(recordings,saving_dir,saving_name,
                                      probe_path,excluded_sites,Plotting=True,
                                      freq_min=300, freq_max=6000,
                                      MOCAP_200Hz_notch=True,
                                      remove_stim_artefact=True)
