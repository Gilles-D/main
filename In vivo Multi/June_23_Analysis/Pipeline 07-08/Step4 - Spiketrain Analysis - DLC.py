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

import spikeinterface as si
import spikeinterface.extractors as se 
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.comparison as sc
import spikeinterface.exporters as sexp
import spikeinterface.widgets as sw
from spikeinterface.curation import MergeUnitsSorting, get_potential_auto_merge

from neo.core import SpikeTrain
from quantities import ms, s, Hz

from elephant.spike_train_generation import homogeneous_poisson_process, homogeneous_gamma_process
from elephant.statistics import mean_firing_rate
from elephant.statistics import time_histogram, instantaneous_rate
from elephant.kernels import GaussianKernel

import pickle
import time
import sys




#%% Parameters
session_name = '0026_29_07'
spikesorting_results_path = r"D:\ePhy\SI_Data\spikesorting_results"
concatenated_signals_path = r'D:\ePhy\SI_Data\concatenated_signals'

sorter_name = "kilosort3"

sorter_folder = rf'{spikesorting_results_path}/{session_name}/{sorter_name}'
signal_folder = rf'{concatenated_signals_path}/{session_name}'




#%%Functions

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
    try:
        # Read the metadata file created during concatenation
        print("Reading the ttl_idx file in intan folder...")
        path = rf'{concatenated_signals_path}/{session_name}/'
        metadata = pickle.load(open(rf"{path}/ttl_idx.pickle", "rb"))
    except:
        print("No recording infos found in the intan folder. Please run Step 0")
    
    # save_path = rf'{spikesorting_results_path}/{session_name}/recordings_info.pickle'
    # if os.path.exists(save_path):
    #     print("Recordings info file exists")
    #     print("Loading info file...")
    #     recordings_info = pickle.load(open(save_path, "rb"))
    # else:
    #     print("Recordings info file does not exist")
    #     print("Getting info...")
    #     # Read the metadata file created during concatenation
    #     path = rf'{concatenated_signals_path}/{session_name}/'
    #     metadata = pickle.load(open(rf"{path}/ttl_idx.pickle", "rb"))
       
    #     # Loop over intan files
    #     recordings_list = metadata['recordings_files']
    #     # RHD file reading
    #     multi_recordings, recordings_lengths, multi_stim_idx, multi_frame_idx, frame_start_delay = [], [], [], [], []
        
    #     # Concatenate recordings
    #     for record in recordings_list:
    #         reader = read_data(record)
    #         signal = reader['amplifier_data']
    #         recordings_lengths.append(len(signal[0]))
    #         multi_recordings.append(signal)
            
    #         stim_idx = reader['board_dig_in_data'][0]  # Digital data for stim of the file
    #         multi_stim_idx.append(stim_idx)  # Digital data for stim of all the files
            
    #         frame_idx = reader['board_dig_in_data'][1]  # Get digital data for mocap ttl
    #         multi_frame_idx.append(frame_idx)  # Digital data for mocap ttl of all the files
            
    #     anaglog_signal_concatenated = np.hstack(multi_recordings)  # Signal concatenated from all the files
    #     digital_stim_signal_concatenated = np.hstack(multi_stim_idx)  # Digital data for stim concatenated from all the files
    #     digital_mocap_signal_concatenated = np.hstack(multi_frame_idx)
        
    #     # Get sampling freq
    #     sampling_rate = reader['frequency_parameters']['amplifier_sample_rate']
        
    #     recordings_lengths_cumsum = np.cumsum(np.array(recordings_lengths) / sampling_rate)
                                              
    #     # Return: recording length, recording length cumsum, digital signals 1 and 2 (in full or logical?)
    #     # Save them in a pickle
        
    #     recordings_info = {
    #         'recordings_length': recordings_lengths,
    #         'recordings_length_cumsum': recordings_lengths_cumsum,
    #         'sampling_rate': sampling_rate,
    #         'digital_stim_signal_concatenated': digital_stim_signal_concatenated,
    #         'digital_mocap_signal_concatenated': digital_mocap_signal_concatenated
    #     }
        
    #     pickle.dump(recordings_info, open(save_path, "wb"))
        
    print('Done')
    return metadata

def plot_waveform(session_name, spikesorting_results_path, sites_location, unit, save=True):
    file_pattern = rf"Unit_{unit} *"
    matching_files = glob.glob(rf"{spikesorting_results_path}/{session_name}/waveforms/{file_pattern}")
    
    print(rf"{spikesorting_results_path}/{session_name}/waveforms/{file_pattern}")
    
    if len(matching_files) > 0:
        print("Matching file(s) found:")
        for file_path in matching_files:
            print(file_path)
            
            df = pd.read_excel(file_path)
            max_wave = df.abs().max().max()  # Calculate the maximum value in all channels
            
            fig1 = plt.figure(figsize=(10, 12))
            ax1 = fig1.add_subplot(111)
            
            fig1.suptitle(rf'Average Waveform Unit # {unit}')
            ax1.set_xlabel('Probe location (micrometers)')
            ax1.set_ylabel('Probe location (micrometers)')
            
            for loc, prob_loc in enumerate(sites_location):
                x_offset, y_offset = prob_loc[0], prob_loc[1]
                base_x = np.linspace(-15, 15, num=len(df.iloc[:, loc]))  # Basic x-array for plot, centered
                # clust_color = 'C{}'.format(cluster)
                
                wave = df.iloc[:, loc] * 100 + max_wave * y_offset  # Adjust y_offset with the max_wave value
                ax1.plot(base_x + 2 * x_offset, wave)
                # ax1.fill_between(base_x + 2 * x_offset, wave - wf_rms[cluster + delta], wave + wf_rms[cluster + delta], alpha=wf_alpha)
                
            plt.show()
            if save == True:
                save_path = rf"{spikesorting_results_path}/{session_name}/plots/waveforms/"
                Check_Save_Dir(save_path)
                plt.savefig(rf"{save_path}/Units {unit}.{plot_format}")
    else:
        print("No matching file found")
        
    return



def create_time_periods(times):
    time_periods = []
    if len(times) < 2:
        return time_periods
    t_start = times[0]
    for i in range(1, len(times)):
        t_stop = times[i]
        time_periods.append((t_start, t_stop))
        t_start = t_stop
    return time_periods


def separate_events_by_period(events, time_periods):
    num_periods = len(time_periods)
    event_periods = []
    
    for i in range(num_periods):
        t_start, t_stop = time_periods[i]
        period_events = events[(events >= t_start) & (events < t_stop)]
        event_periods.append(period_events - t_start)
        
    event_periods_dict = {}
    
    for i, array in enumerate(event_periods):
        event_periods_dict[rf'Mocap_Session_{i + 1}'] = array
        
    return event_periods_dict


def plot_heatmap(spike_times, bin_size):

    event_times_list = spike_times.values()
    event_session_list = spike_times.keys()
    
    # Set the bin size and create the time bins
    max_time = max(max(times) for times in event_times_list)
    
    bins = np.arange(0, max_time + bin_size, bin_size)
    
    # Compute the histograms of event counts for each row
    histograms = []
    for event_times in event_times_list:
        counts, _ = np.histogram(event_times, bins)
        histograms.append(counts)
        
    # Check if any histogram has fewer than two bins
    if any(len(counts) < 2 for counts in histograms):
        raise ValueError("Insufficient data for heatmap visualization.")
        
    # Create a 2D array from the histograms for heatmap visualization
    heatmap = np.array(histograms)
    
    # Plot the heatmap
    plt.figure()
    plt.imshow(heatmap, cmap='hot', aspect='auto')
    plt.colorbar(label='Event Count')
    plt.xlabel('Time Bin')
    plt.ylabel('Event session')
    
    # Set the y-axis tick labels to the event_session_list items
    plt.yticks(range(len(event_session_list)), event_session_list)
    
    plt.title('Event Heatmap in Line Plot')
    plt.show()




def plot_heatmap_start_fixed(spike_times, bin_size):

    event_times_list = spike_times.values()
    event_session_list = spike_times.keys()
    
    # Set the bin size and create the time bins
    max_time = max(max(times) for times in event_times_list)
    
    bins = np.arange(-10, max_time + bin_size, bin_size)
    
    # Compute the histograms of event counts for each row
    histograms = []
    for event_times in event_times_list:
        counts, _ = np.histogram(event_times, bins)
        histograms.append(counts)
        
    # Check if any histogram has fewer than two bins
    if any(len(counts) < 2 for counts in histograms):
        raise ValueError("Insufficient data for heatmap visualization.")
        
    # Create a 2D array from the histograms for heatmap visualization
    heatmap = np.array(histograms)
    
    # Plot the heatmap
    
    plt.imshow(heatmap, cmap='hot', aspect='auto')
    plt.colorbar(label='Event Count')
    plt.xlabel('Time Bin')
    plt.ylabel('Event session')
    
    # Set the y-axis tick labels to the event_session_list items
    plt.yticks(range(len(event_session_list)), event_session_list)
    
    plt.show()



def find_start_stop_obstacle(mocap_folder,session_name,mocap_frequency=200):
    list_mocap_files = glob.glob(os.path.join(rf"{mocap_folder}/{session_name}", '*.csv'))
    mocap_data_dict={}
    
    for file in list_mocap_files:
        print(file)
        data_MOCAP = MA.MOCAP_file(file)
        session_idx = int(file.split('\\')[-1].split('.')[0].split('_')[-1])
        session = rf"Mocap_Session_{session_idx}"
        
        back_coord = data_MOCAP.coord(f"{data_MOCAP.subject()}:Back1")
        start_x = -np.nanmedian(data_MOCAP.coord(f"{data_MOCAP.subject()}:Platform1")[1])
        stop_x = -np.nanmedian(data_MOCAP.coord(f"{data_MOCAP.subject()}:Platform2")[1])
           
       
        start_frame = np.where(-back_coord[1] > start_x)[0][0]
        start_time = start_frame/mocap_frequency
        
        stop_frame = np.where(-back_coord[1] > stop_x)[0][0]
        stop_time = stop_frame/mocap_frequency
        
        try:
            obstacle_x = -np.nanmedian(data_MOCAP.coord(f"{data_MOCAP.subject()}:Obstacle1")[1])
            obstacle_frame = np.where(-back_coord[1] > obstacle_x)[0][0]
            obstacle_time = obstacle_frame/mocap_frequency
            
        except:
            print("no obstacle")
            obstacle_time=np.nan
        
        
        
        # plt.figure()
        # plt.title(file)
        # plt.plot(-back_coord[1],back_coord[2])
        # plt.axvline(start_x)
        # plt.axvline(stop_x)
        # plt.axvline(obstacle_x)
        
        
        file_dict = {
            "start_time": start_time,
            "stop_time": stop_time,
            "obstacle_time": obstacle_time
        }
        
        mocap_data_dict[session] = file_dict
    return mocap_data_dict



def get_coord(filepath, bodypart):
    df = pd.read_csv(filepath)
    
    col_x = df.columns[(df.iloc[0] == bodypart) & (df.iloc[1] == "x")]
    col_y = df.columns[(df.iloc[0] == bodypart) & (df.iloc[1] == "y")]
    
    # Sélectionner la colonne correspondante à l'aide de loc
    selected_colx = df.loc[:, col_x[0]].values
    selected_coly = df.loc[:, col_y[0]].values
    
    return selected_colx[2:].astype(float), selected_coly[2:].astype(float)
       
def calculer_vitesse(x, y, dt):
    dx = np.diff(x)
    dy = np.diff(y)
    vitesse = np.sqrt(dx**2 + dy**2) / dt
    return vitesse

def distance_entre_points(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)




#%% Loadings
recordings_info = Get_recordings_info(session_name,concatenated_signals_path,spikesorting_results_path)

sorter_results = ss.NpzSortingExtractor.load_from_folder(rf'{sorter_folder}/curated').remove_empty_units()
signal = si.load_extractor(signal_folder)
we = si.WaveformExtractor(signal,sorter_results)

sampling_rate = 20000*Hz
time_axis = signal.get_times()




#%% 1 - Whole session spiketrain analysis
Plot_histogram = False

print('Figure 1 - Elephant Spike Train Analysis')

unit_list = sorter_results.get_num_units()
unit_index_list = sorter_results.get_unit_ids()

for unit in range(unit_list):
    
    unit_id = unit_index_list[unit]
 
    spiketrain = SpikeTrain(sorter_results.get_unit_spike_train(unit_id=unit_id)/sampling_rate, t_stop=time_axis[-1])
    
    
    print(rf"The mean firing rate of unit {unit_id} on whole session is", mean_firing_rate(spiketrain))
    
    plt.figure()    
    
    histogram_count = time_histogram([spiketrain], 0.5*s)
    histogram_rate = time_histogram([spiketrain],  0.5*s, output='rate')
    
    #Histogram info
    """
    print(type(histogram_count), f"of shape {histogram_count.shape}: {histogram_count.shape[0]} samples, {histogram_count.shape[1]} channel")
    print('sampling rate:', histogram_count.sampling_rate)
    print('times:', histogram_count.times)
    print('counts:', histogram_count.T[0])
    print('times:', histogram_rate.times)
    print('rate:', histogram_rate.T[0])
    """   
    
    
    inst_rate = instantaneous_rate(spiketrain, sampling_period=50*ms)
    
    #instantaneous rate info
    """
    print(type(inst_rate), f"of shape {inst_rate.shape}: {inst_rate.shape[0]} samples, {inst_rate.shape[1]} channel")
    print('sampling rate:', inst_rate.sampling_rate)
    print('times (first 10 samples): ', inst_rate.times[:10])
    print('instantaneous rate (first 10 samples):', inst_rate.T[0, :10])
    """

    # plotting the original spiketrain
    # plt.plot(spiketrain, [0]*len(spiketrain), 'r', marker=2, ms=25, markeredgewidth=2, lw=0, label='poisson spike times')
    
    
    
    # time histogram
    if Plot_histogram == True:
        
        plt.bar(histogram_rate.times, histogram_rate.magnitude.flatten(), width=histogram_rate.sampling_period,
                align='edge', alpha=0.3, label='time histogram (rate)',color='black')
    
    
    
    # Instantaneous rate
    plt.plot(inst_rate.times.rescale(s), inst_rate.rescale(histogram_rate.dimensionality).magnitude.flatten(), label='instantaneous rate')
    
    
    # axis labels and legend
    plt.xlabel('time [{}]'.format(spiketrain.times.dimensionality.latex))
    plt.ylabel('firing rate [{}]'.format(histogram_rate.dimensionality.latex))
    
    plt.xlim(spiketrain.t_start, spiketrain.t_stop)   
    #plt.xlim(0, 572.9232) #Use this to focus on phases you want using recordings_lengths_cumsum
    
    
    plt.legend()
    plt.title(rf'Unit #{unit_id}')
    plt.show()
    


#%% DLC movement bouts from Openfield task
#TODO : load all DLC files of the session (give the list in parameters ?)

#TODO : align DLC data with TTL starts

def get_DLC_mouvement_bouts(filepath,dlc_meta_filepath,LED_delay_frames, recordings_info,
                            sampling_rate=20000, plot_trajectory=True,plot_speed=True,
                            plot_speed_bouts=True,
                            ):
    
    TTL_idx = (np.where(recordings_info['digital_mocap_signal_concatenated'] == True))[0]
    video_starts_times = start_TTL_detection(TTL_idx, sampling_rate)
    
    with open(dlc_meta_filepath, 'rb') as handle:
        DLC_meta = pickle.load(handle)
    
    freq=DLC_meta['data']['fps']
    LED_delay_time = LED_delay_frames/freq
    
    
    x,y = get_coord(filepath, "Tail_base")
    time_axis_DLC =np.array(range(len(x)))/freq
    
    
    # Ajouter cette condition pour enlever les points entre 0 et 5
    # Souvent un bug de marqueur qui saute à l'origine
    indices_a_supprimer = np.where((x >= 0) & (x <= 5))[0]   
    
    # Calculer la distance entre les points consécutifs
    distances = distance_entre_points(x[:-1], y[:-1], x[1:], y[1:])
    
    # Définir un seuil pour la distance au-delà duquel on considère qu'il y_smooth a un artefact
    seuil_distance = 80
    
    # Indiquer les indices des points où la distance dépasse le seuil (ce sont les artefacts)
    indices_artefacts_distance = np.where(distances > seuil_distance)[0] + 1
    
    # Indiquer les indices des points où la vitesse dépasse le seuil (ce sont les artefacts, comme précédemment)
    dx_smooth = np.diff(x)
    dy_smooth = np.diff(y)
    vitesse = np.sqrt(dx_smooth**2 + dy_smooth**2)
    seuil_vitesse = 80
    indices_artefacts_vitesse = np.where(vitesse > seuil_vitesse)[0]
    
    # Fusionner les indices d'artefacts basés sur la vitesse et la distance
    indices_artefacts_total = np.union1d(indices_artefacts_distance, indices_artefacts_vitesse)
    indices_artefacts_total = np.union1d(indices_artefacts_total, indices_a_supprimer)
    
    # Supprimer les points correspondants aux_smooth indices des artefacts de la trajectoire
    x_corrige = np.delete(x, indices_artefacts_total)
    y_corrige = np.delete(y, indices_artefacts_total)
    time_axis_corrige = np.delete(time_axis_DLC, indices_artefacts_total)
    
    if plot_trajectory == True: 
        # Plot de la trajectoire brute et de la trajectoire corrigée
        plt.figure()
        # plt.plot(x, y, 'o-', label='Trajectoire brute')
        plt.plot(x_corrige, y_corrige, 'r-', label='Trajectoire corrigée')
        plt.legend()
        plt.xlabel('Position x')
        plt.ylabel('Position y')
        plt.title('Correction des artefacts dans la trajectoire')
        plt.grid(True)
        plt.gca().invert_yaxis()
        plt.show()
    
    
    # Exemple de données de vitesse (simulées)
    time = time_axis_corrige[1:]  # Temps
    velocity = calculer_vitesse(x_corrige,y_corrige,dt=1/freq)  # Vitesse (simulée)
    
    # Définir le seuil de vitesse pour distinguer l'immobilité du mouvement
    seuil_vitesse = 20
    
    # Identifier les indices où la vitesse est supérieure au seuil (mouvement)
    indices_mouvement = np.where(velocity > seuil_vitesse)[0]
    
    # Identifier les indices où la vitesse est inférieure ou égale au seuil (immobilité)
    indices_immobilite = np.where(velocity <= seuil_vitesse)[0]
    
    if plot_speed == True:
        
        # Créer un plot montrant la vitesse et les périodes d'immobilité et de mouvement
        plt.figure(figsize=(10, 6))
        plt.plot(time, velocity, label='Vitesse')
        plt.plot(time[indices_immobilite], velocity[indices_immobilite], 'ro', label='Immobilite')
        plt.plot(time[indices_mouvement], velocity[indices_mouvement], 'go', label='Mouvement')
        plt.axhline(y=seuil_vitesse, color='gray', linestyle='--', label='Seuil')
        plt.xlabel('Temps')
        plt.ylabel('Vitesse')
        plt.legend()
        plt.title('Décomposition des phases d\'immobilité et de mouvement')
        plt.show()
    
    # Définir la taille de la fenêtre glissante
    taille_fenetre = 10
    
    # Créer un masque avec True pour les points au-dessus du seuil et False pour les points en-dessous du seuil
    masque_mouvement = velocity > seuil_vitesse
    
    # Utiliser une fonction de convolution pour identifier les périodes de mouvement
    convolution = np.convolve(masque_mouvement, np.ones(taille_fenetre), mode='valid')
    
    # Identifier les indices où la convolution est égale à la taille de la fenêtre (c'est-à-dire où les 10 points consécutifs sont au-dessus du seuil)
    indices_mouvement = np.where(convolution == taille_fenetre)[0]
    
    # Identifier les indices où la convolution est égale à zéro (c'est-à-dire où les 10 points consécutifs sont en-dessous ou égaux au seuil)
    indices_immobilite = np.where(convolution == 0)[0]
    
    if plot_speed_bouts == True:
        # Créer un plot montrant la vitesse et les périodes d'immobilité et de mouvement
        plt.figure(figsize=(10, 6))
        plt.plot(time, velocity, label='Vitesse')
        plt.plot(time[indices_immobilite], velocity[indices_immobilite], 'ro', label='Immobilite')
        plt.plot(time[indices_mouvement], velocity[indices_mouvement], 'go', label='Mouvement')
        plt.axhline(y=seuil_vitesse, color='gray', linestyle='--', label='Seuil')
        plt.xlabel('Temps')
        plt.ylabel('Vitesse')
        plt.legend()
        plt.title('Décomposition des phases d\'immobilité et de mouvement')
        plt.show()
    
    
    # Déterminer les phases d'immobilité sous forme d'un tableau de tuples (début, fin)
    phases_immobilite = []
    
    # Initialiser les indices de début et de fin de la phase d'immobilité
    debut_phase = indices_immobilite[0]/freq
    fin_phase = indices_immobilite[0]/freq
    
    # Parcourir les indices d'immobilité pour regrouper les périodes consécutives en une seule phase
    for i in range(1, len(indices_immobilite)):
        if indices_immobilite[i] == indices_immobilite[i-1] + 1:
            # Si l'indice est consécutif à l'indice précédent, il fait toujours partie de la même phase
            fin_phase = indices_immobilite[i]
        else:
            # Sinon, nous avons trouvé la fin de la phase d'immobilité précédente et nous devons enregistrer cette phase
            phases_immobilite.append((debut_phase/freq, fin_phase/freq))
            # Déplacer les indices de début et de fin pour la prochaine phase
            debut_phase = indices_immobilite[i]
            fin_phase = indices_immobilite[i]
            
    # Ajouter la dernière phase d'immobilité au tableau
    phases_immobilite.append((debut_phase/freq, fin_phase/freq))
    
    delta = video_starts_times[1]-LED_delay_time
    phases_immobilite = [(x + delta, y + delta) for x, y in phases_immobilite]
       
    return phases_immobilite


phases_immobilite = get_DLC_mouvement_bouts("D:/Videos/0012/shuffle 2/0026_29_07_01DLC_resnet50_OpenfieldJul31shuffle2_200000_filtered.csv",
                               dlc_meta_filepath=fr'D:/Videos/0012/shuffle 2/0026_29_07_01DLC_resnet50_OpenfieldJul31shuffle2_200000_meta.pickle',
                               LED_delay_frames=40,
                               recordings_info=recordings_info)



#%%Extract stim TTL idx
TTL_stim_idx = (np.where(recordings_info['digital_stim_signal_concatenated'] == True))[0]
stim_start_times = start_TTL_detection(TTL_stim_idx, sampling_rate)

stim_start_idx = stim_start_times*sampling_rate

#Save stim times
with open(rf'{spikesorting_results_path}\{session_name}\stim_times_dict.pkl', 'wb') as f:
    pickle.dump(stim_start_times, f)
    
#Save stim idx
with open(rf'{spikesorting_results_path}\{session_name}\stim_idx_dict.pkl', 'wb') as f:
    pickle.dump(stim_start_idx, f)





#%% Figure 2 - Heatmap spiking by unit, by mocap session

# Create the save directory
savefig_folder = rf'{spikesorting_results_path}/{session_name}/plots/heatmaps/'
Check_Save_Dir(savefig_folder)

for unit in spike_times_dict['Units']:
    print(unit)
    spike_times = spike_times_by_mocap_session[unit]
    plot_heatmap(spike_times,bin_size = 0.5)
    plt.savefig(rf"{savefig_folder}{unit}.png")
    
    
    # t_stop = None

    # for key, value in spike_times.items():
    #     if t_stop is None:
    #         t_stop = np.max(value)
    #     else:
    #         t_stop = max(t_stop, np.max(value))
    
    # spiketrains=[]
    
    # # plt.figure()
    # # plt.title(rf'Unit # {unit}')
    
    # for i,session in enumerate(spike_times.keys()):
        
    #     spiketrain = SpikeTrain(spike_times[session]*s, t_stop)
    #     inst_rate = instantaneous_rate(spiketrain, 5*ms)
        

#%% 3- Raster + Psth start
# Create the save directory
savefig_folder = rf'{spikesorting_results_path}/{session_name}/plots/Raster-PSTH/'
Check_Save_Dir(savefig_folder)

with open(fr'{spikesorting_results_path}/{session_name}/spike_times_by_mocap_session.pickle', 'rb') as handle:
    spike_times_by_mocap_session = pickle.load(handle)

for unit in spike_times_dict['Units']:
    print(unit)
    spike_times = spike_times_by_mocap_session[unit]
    for session, start_stop_dict in mocap_start_stop_dict.items():
        start_time = start_stop_dict['start_time']
        if session in spike_times:
            spike_times[session] -= start_time
    
    # Créer une figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    
    # Sous-graphique supérieur pour les spike_times
    ax1.set_title(rf'Unit {unit}')
    for i, (session, session_times) in enumerate(spike_times.items()):
        color = 'black' if session_times[i] < 0 else 'red'
        ax1.eventplot([session_times], lineoffsets=i + 0.5, linelengths=0.5, color=color)
    ax1.axvline(0, color='red')
    ax1.set_xlim(-10, 15)
    ax1.set_ylim(0, 11)
    
    # Sous-graphique inférieur pour le PSTH
    ax2.set_xlabel('Temps (s)')
    ax2.set_ylabel('Nombre de spikes')
    ax2.set_title('PSTH des 11 premières sessions')
    
    session_times = np.concatenate(list(spike_times.values())[:11])
    bin_size = 0.01  # Taille de chaque bin en secondes
    num_bins = int((max(session_times) - min(session_times)) / bin_size)
    
    hist, bin_edges = np.histogram(session_times, bins=num_bins, range=(min(session_times), max(session_times)))
    
    bar_color = np.where(bin_edges[:-1] < 0, 'black', 'red')
    ax2.bar(bin_edges[:-1], hist, width=bin_size, color=bar_color)
    
    # Ajuster l'espace entre les deux sous-graphiques
    plt.subplots_adjust(hspace=0.3)
    
    # Enregistrer la figure
    plt.savefig(rf"{savefig_folder}{unit}.png")
    
    # Afficher la figure
    plt.show()

#%%Raster + PSTH obstacle
# Create the save directory
savefig_folder = rf'{spikesorting_results_path}/{session_name}/plots/Raster-PSTH-obstacle/'
Check_Save_Dir(savefig_folder)

with open(fr'{spikesorting_results_path}/{session_name}/spike_times_by_mocap_session.pickle', 'rb') as handle:
    spike_times_by_mocap_session = pickle.load(handle)

for unit in spike_times_dict['Units']:
    print(unit)
    spike_times = spike_times_by_mocap_session[unit]
    for session, start_stop_dict in mocap_start_stop_dict.items():
        obstacle_time = start_stop_dict['obstacle_time']
        if session in spike_times:
            spike_times[session] -= obstacle_time
    
    # Créer une figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    
    # Sous-graphique supérieur pour les spike_times
    ax1.set_title(rf'Unit {unit}')
    for i, (session, session_times) in enumerate(spike_times.items()):
        color = 'black' if session_times[i] < 0 else 'red'
        ax1.eventplot([session_times], lineoffsets=i + 0.5, linelengths=0.5, color=color)
    ax1.axvline(0, color='red')
    ax1.set_xlim(-10, 15)
    ax1.set_ylim(7, 11)
    
    # Sous-graphique inférieur pour le PSTH
    ax2.set_xlabel('Temps (s)')
    ax2.set_ylabel('Nombre de spikes')
    ax2.set_title('PSTH des sessions obstacles')
    
    session_times = np.concatenate(list(spike_times.values())[7:10])
    bin_size = 0.01  # Taille de chaque bin en secondes
    num_bins = int((max(session_times) - min(session_times)) / bin_size)
    
    hist, bin_edges = np.histogram(session_times, bins=num_bins, range=(min(session_times), max(session_times)))
    
    bar_color = np.where(bin_edges[:-1] < 0, 'black', 'red')
    ax2.bar(bin_edges[:-1], hist, width=bin_size, color=bar_color)
    
    # Ajuster l'espace entre les deux sous-graphiques
    plt.subplots_adjust(hspace=0.3)
    
    # Enregistrer la figure
    plt.savefig(rf"{savefig_folder}{unit}.png")
    
    # Afficher la figure
    plt.show()





    # spike_times = spike_times_by_mocap_session[unit]
    # for session, start_stop_dict in mocap_start_stop_dict.items():
    #     start_time = start_stop_dict['start_time']
    #     if session in spike_times:
    #         spike_times[session] -= start_time
    
    # # Créer une figure avec deux sous-graphiques
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    
    # # Sous-graphique supérieur pour les spike_times
    # ax1.set_title(rf'Unit {unit}')
    # ax1.eventplot(spike_times.values())
    # ax1.axvline(0)
    # ax1.set_xlim(-10, 15)
    # ax1.set_ylim(0, 10)
    
    # # Sous-graphique inférieur pour le PSTH
    # ax2.set_xlabel('Temps (s)')
    # ax2.set_ylabel('Nombre de spikes')
    # ax2.set_title('PSTH des 10 premières sessions')
    
    # ax2.axvline(0)
    
    # session_times = np.concatenate(list(spike_times.values())[:10])
    # bin_size = 0.01  # Taille de chaque bin en secondes
    # num_bins = int((max(session_times) - min(session_times)) / bin_size)
    
    # hist, bin_edges = np.histogram(session_times, bins=num_bins, range=(min(session_times), max(session_times)))
    
    # ax2.bar(bin_edges[:-1], hist, width=bin_size)
    
    # # Ajuster l'espace entre les deux sous-graphiques
    # plt.subplots_adjust(hspace=0.3)
    
    # # Enregistrer la figure
    # plt.savefig(rf"{savefig_folder}Raster_PSTH_{unit}.png")
    
    # # Afficher la figure
    # plt.show()
    
            
#%% Optotag
#Select only optotag session with minimum idx

idx_optotag_min = recordings_info['stim_ttl_on'][1325]

#Select stim idx greater than this limit
selected_stim_idx = recordings_info['stim_ttl_on'][recordings_info['stim_ttl_on'] > idx_optotag_min]
selected_stim_times = selected_stim_idx/sampling_rate

optotag_data = []

for idx,unit in enumerate(spike_times_dict['Units']):
    #Raster with the stim
    spike_times = spike_times_dict['spike times'][idx]
    first_event = []
    
    # Définir la fenêtre temporelle
    fenetre_avant = 0.05  # 100 ms avant
    fenetre_apres = 0.05   # 100 ms après
    
    # Créer une figure
    plt.figure(figsize=(10, 6))
    
    # Créer un tableau pour stocker les positions des événements
    event_positions = []
    
    # Parcourir les temps de stimulation
    for idx, stimulation in enumerate(selected_stim_times):
        # Sélectionner les événements dans la fenêtre autour du temps de stimulation
        evenements_autour = spike_times[(spike_times >= stimulation - fenetre_avant) & 
                                        (spike_times <= stimulation + fenetre_apres)]
        
        # Calculer les positions relatives des événements par rapport à la stimulation
        relative_positions = evenements_autour - stimulation
        
        # Ajouter les positions relatives au tableau
        event_positions.extend(relative_positions)
        
        # Trouver le premier événement après la stimulation
        if len(relative_positions[relative_positions > 0]) > 0:
            
            first_event.append(relative_positions[relative_positions > 0][0])
        else:
            first_event.append(None)  # Aucun événement après la stimulation
    
    # Créer un histogramme des réponses
    bins = np.arange(-fenetre_avant, fenetre_apres + 0.001, 0.001)  # Pas de 1 ms
    plt.hist(event_positions, bins=bins, color='b', alpha=0.7)
    
    # Marquer les temps de stimulation
    plt.vlines(0, 0, plt.gca().get_ylim()[1], color='r', label='Stimulation')
    
    # Étiquettes et titre
    plt.xlabel('Temps (s) par rapport à la stimulation')
    plt.ylabel('Nombre d\'événements')
    plt.title(rf'Optotag raster {unit}')
    plt.legend()
    
    # Afficher la figure
    plt.show()
    event_positions_array = np.array(event_positions)
   
    reliability = first_event.count(None)/len(first_event)*100
    delay = np.nanmean(np.array(first_event, dtype=float))
    jitter = np.nanstd(np.array(first_event, dtype=float)) / np.sqrt(np.sum(~np.isnan(np.array(first_event, dtype=float))))*1000
    
    print(rf'{unit} Reliability = {reliability} % | Delay = {jitter} +/- {jitter} ms ')


    optotag_data.append((unit, np.array([x for x in first_event if x is not None])))
    
data_dict = {unit: first_event for unit, first_event in optotag_data}

# Créer un DataFrame à partir du dictionnaire
data_frame = pd.DataFrame.from_dict(data_dict, orient='index').T

# Utiliser Seaborn pour créer le box plot horizontal
plt.figure(figsize=(10, 6))
# sns.boxplot(data=data_frame, orient='h')
# sns.stripplot(data=data_frame, orient='h',color='black')
sns.violinplot(data=data_frame)

# Ajouter des étiquettes aux axes
plt.xlabel('Temps (ms)')
plt.ylabel('Noms')
plt.title('Box Plot Horizontal des Temps d\'Apparition d\'Événements')

# Afficher le graphique
plt.show()