# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 14:36:03 2023

@author: Gil
"""

import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neo.core import SpikeTrain
from quantities import ms, s, Hz
from elephant.spike_train_generation import homogeneous_poisson_process, homogeneous_gamma_process
from elephant.statistics import mean_firing_rate
from elephant.statistics import time_histogram, instantaneous_rate
from elephant.kernels import GaussianKernel

import pickle


#%% Parameters
session_name = r'0012_12_07_allfiles_allchan'
working_directory = r"D:\Seafile\Seafile\Data\ePhy\2_SI_data\spikesorting_results"

plot_format = 'png'




#%%Functions
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

def Get_spikes(session_name):
    """
    Retrieves spike times from XLSX files in a given session directory.

    Args:
        session_name (str): Name of the session directory.

    Returns:
        dict: Dictionary containing unit names, spike times array, and spike trains.

    """

    # Set the path to the spike times directory
    spike_times_path = rf'D:\Seafile\Seafile\Data\ePhy\2_SI_data\spikesorting_results\{session_name}\spikes'

    # List all units
    unit_list = [file_name.split(".")[0] for file_name in os.listdir(spike_times_path)]

    # Use glob.glob() to get a list of XLSX files in the directory
    file_paths = glob.glob(os.path.join(spike_times_path, '*.xlsx'))

    # Create lists to store the spike times arrays and spike trains
    spike_times_array, elephant_spiketrains = [], []

    # Loop through the XLSX files
    for file_path in file_paths:
        # Load the XLSX file into a pandas DataFrame and retrieve the second column as spike times
        spike_times = np.array(pd.read_excel(file_path).iloc[:, 1])
        spike_times_array.append(spike_times)

        # Calculate t_stop as the maximum spike time plus 1
        t_stop = max(spike_times) + 1

        # Create a spike train using the Elephant library
        elephant_spiketrains.append(SpikeTrain(spike_times * s, t_stop=t_stop))

    # Create a dictionary containing the unit names, spike times arrays, and spike trains
    spike_times_dict = {'Units': unit_list, 'spike times': spike_times_array, 'spiketrains': elephant_spiketrains}

    return spike_times_dict


def Get_recordings_info(session_name):
    #Check si le fichier existe déja et le load si c'est le cas
    
    #lire le fichier metadata créée lors du concatenate
    
    #boucle intan files
    
    #return : recording length, recording length cumsum, signaux digitaux 1 et 2 (en full ou logique ?)
    #les sauvegarde dans un pickle

    return



#%%
spike_times_dict = Get_spikes(session_name)



#%% Figure 1 : Whole SpikeTrain Analysis
print('Figure 1 - Elephant Spike Train Analysis')

data = pickle.load(open("D:/Seafile/Seafile/Data/ePhy/2_SI_data/concatenated_signals/0012_03_07_allfiles_allchan/concatenated_recording_trial_time_index_df.pickle", "rb"))

for unit in spike_times_dict['Units']:
    print(unit)
    spiketrain = spike_times_dict['spiketrains'][spike_times_dict['Units'].index(unit)]

    print(rf"The mean firing rate of unit {unit} on whole session is", mean_firing_rate(spiketrain))
    
    plt.figure()    

    histogram_count = time_histogram([spiketrain], 0.5*s)
    histogram_rate = time_histogram([spiketrain],  0.5*s, output='rate')  
    
    inst_rate = instantaneous_rate(spiketrain, sampling_period=50*ms)
    

    # plotting the original spiketrain (rasterplot)
    # plt.plot(spiketrain, [0]*len(spiketrain), 'r', marker=2, ms=25, markeredgewidth=2, lw=0, label='poisson spike times')
    
    
    # Mean firing rate for the baseline phase
    # baseline_stop = baseline_duration
    # plt.hlines(mean_firing_rate(spiketrain,t_stop=baseline_stop*s), xmin=spiketrain.t_start, xmax=spiketrain.t_stop, linestyle='--', label='mean firing rate')
    
    
    # time histogram
    plt.bar(histogram_rate.times, histogram_rate.magnitude.flatten(), width=histogram_rate.sampling_period,
            align='edge', alpha=0.3, label='time histogram (rate)',color='black')
       
    # Instantaneous rate
    plt.plot(inst_rate.times.rescale(s), inst_rate.rescale(histogram_rate.dimensionality).magnitude.flatten(), label='instantaneous rate')
    
    #Length of each recordings
    # [plt.axvline(_x, linewidth=1, color='g') for _x in recordings_lengths_cumsum]
    
    #Mocap ttl
    # [plt.axvline(_x, linewidth=1, color='b') for _x in mocap_starts_times]
    
    
    
    # axis labels and legend
    plt.xlabel('time [{}]'.format(spiketrain.times.dimensionality.latex))
    plt.ylabel('firing rate [{}]'.format(histogram_rate.dimensionality.latex))
    
    plt.xlim(spiketrain.t_start, spiketrain.t_stop)   
    #plt.xlim(0, 572.9232) #Use this to focus on phases you want using recordings_lengths_cumsum
    
    
    plt.legend()
    plt.title(rf'Spiketrain {unit}')
    plt.show()
    
    # Check_Save_Dir(savefig_folder)
    savefig_folder = rf'{working_directory}/{session_name}/plots/spiking_analysis/'
    Check_Save_Dir(savefig_folder)
    plt.savefig(rf"{savefig_folder}/Figure 1 - Elephant Spike Train Analysis - {unit}.{plot_format}")