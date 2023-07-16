import spikeinterface as si  # import core only
import spikeinterface.sorters as ss
import spikeinterface.widgets as sw
import spikeinterface.comparison as sc
import spikeinterface.qualitymetrics as sqm
from spikeinterface.curation import CurationSorting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from viziphant.statistics import plot_time_histogram
from viziphant.rasterplot import rasterplot_rates
from elephant.statistics import time_histogram
from neo.core import SpikeTrain
from quantities import s, ms
import numpy as np
import pickle
import os
from pylab import get_current_fig_manager
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

sorter_dict = {
                'paramCustom': {
                                'tridesclous':   {'detect_sign': -1, # "Use -1 (negative) or 1 (positive) depending on the sign of the spikes in the recording"
                                                    'detect_threshold': 5, # "Threshold for spike detection"
                                                },
                                'herdingspikes': {
                                                    'detect_threshold': 20,  # 24, #15, "Detection threshold"
                                                    'filter': False,
                                                },
                                'mountainsort4': {
                                                'detect_sign': -1,  # Use -1, 0, or 1, depending on the sign of the spikes in the recording "Use -1 (negative) or 1 (positive) depending on the sign of the spikes in the recording"
                                                'adjacency_radius': -1,  # Use -1 to include all channels in every neighborhood "Radius in um to build channel neighborhood (Use -1 to include all channels in every neighborhood)"
                                                'freq_min': 300,  # Use None for no bandpass filtering "High-pass filter cutoff frequency"
                                                'freq_max': 6000,# "Low-pass filter cutoff frequency"
                                                'filter': False,# "Enable or disable filter"
                                                'whiten': True,  # Whether to do channel whitening as part of preprocessing "Enable or disable whitening"
                                                'num_workers': None,# "Number of workers (if None, half of the cpu number is used)"
                                                'clip_size': 500,# "Number of samples per waveform"
                                                'detect_threshold': 6,# "Threshold for spike detection"
                                                'detect_interval': 10,  # Minimum number of timepoints between events detected on the same channel "Minimum number of timepoints between events detected on the same channel"
                                                'tempdir': None# "Temporary directory for mountainsort (available for ms4 >= 1.0.2)s"
                                                },
                                },
    }

def plot_maker(sorter, we, save, sorter_name, save_path, delay, mouse):
    for unit_id in sorter.get_unit_ids():
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
        if save:
            plt.savefig(fr'C:\local_data\Paper\Data\spike\{mouse}_{delay}\{sorter_name}\we\Unit_{int(unit_id)}.pdf')
            plt.close()


def run_multiple_sorter(save, nb_of_agreement=0, plot_sorter=True, plot_comp=True):
    # ['173', '174', '176', '6401', '6402', '6409', '6457', '6456', '6924', '6928']
    # for mouse in ['173', '174', '176', '6401', '6402', '6409', '6457', '6456', '6924', '6928']:
    #     print('\n', mouse)
    #     #['Fixed Delay', 'Random Delay']
    #     for delay in ['Fixed_Delay']:
    #         print(delay)
    #         if delay == 'Fixed_Delay':
    #             recording_delay = 'Fixed Delay'
    #         elif delay == 'Random_Delay':
    #             recording_delay = 'Random Delay'
    #         else:
    #             raise ValueError(f'Unrecognize Delay({delay})')
                
            recording_path = fr'C:\local_data\Paper\Data\concaneted_recording\{mouse}_{recording_delay}'
            os.chdir(recording_path)
            
            multirecording = si.BinaryFolderRecording(f'{recording_path}/concatenated_recording')
            multirecording.annotate(is_filtered=True)
            
            print('================================')
            for param_name in sorter_dict.keys():
                sorter_list = []
                sorter_name_list = []
                for sorter_name, sorter_params in sorter_dict[param_name].items():
                    spike_folder = f'C:\local_data\Paper\Data\spike\{mouse}_{delay}\{sorter_name}'
                    print(sorter_name,)
                    print(f'{sorter_name} Start')
                    docker_image = False if sorter_name in ['spykingcircus2', 'tridesclous2'] else True #Those sorter are build in spike interface and dont need dockers
                    if os.path.isdir(f'{spike_folder}\sorter'):
                        print('Sorter folder found, load from folder')
                        sorter_result = ss.NpzSortingExtractor.load_from_folder(f'{spike_folder}\sorter\in_container_sorting')
                    else:
                        print('Sorter folder not found, computing from raw recordings')
                        sorter_result = ss.run_sorter(sorter_name=sorter_name, recording=multirecording, docker_image=docker_image,
                                                    output_folder=f'{spike_folder}\sorter', verbose=True, **sorter_params)

                    sorter_list.append(sorter_result)
                    sorter_name_list.append(sorter_name)
                    
                    
                    #save the sorter params
                    with open(f'{spike_folder}\param_dict.pkl', 'wb') as f:
                        pickle.dump(sorter_params, f)
                    if os.path.isdir(f'{spike_folder}\we'):
                        print('Waveform folder found, load from folder')
                        we = si.WaveformExtractor.load_from_folder(f'{spike_folder}\we', sorting=sorter_result)
                    else:
                        we = si.extract_waveforms(multirecording, sorter_result, folder=f'{spike_folder}\we')

                    if plot_sorter:
                        print('Plot sorting summary in progress')
                        plot_maker(sorter_result, we, save, sorter_name, spike_folder, delay, mouse)
                        print('Plot sorting summary finished')
                    print('================================')

                if len(sorter_list) > 1 and nb_of_agreement != 0:
                    ############################
                    # Sorter outup comparaison #
                    base_comp_folder = f'C:\local_data\Paper\Data\spike\{mouse}_{delay}'
                    comp_multi_name = f'comp_mult_{nb_of_agreement}'
                    for sorter_name in sorter_name_list:
                        comp_multi_name += f'_{sorter_name}'
                    base_comp_folder = f'{base_comp_folder}\{comp_multi_name}'

                    if os.path.isdir(f'{base_comp_folder}\sorter'):
                        print('multiple comparaison sorter folder found, load from folder')
                        sorting_agreement = ss.NpzSortingExtractor.load_from_folder(f'{base_comp_folder}\sorter')
                    else:
                        print('multiple comparaison sorter folder not found, computing from sorter list')
                        comp_multi = sc.compare_multiple_sorters(sorting_list=sorter_list,
                                                                name_list=sorter_name_list)
                        comp_multi.save_to_folder(base_comp_folder)
                        # del sorting_list, sorting_name_list
                        sorting_agreement = comp_multi.get_agreement_sorting(minimum_agreement_count=nb_of_agreement)
                        sorting_agreement.save_to_folder(f'{base_comp_folder}\sorter')
                    try:
                        we = si.extract_waveforms(multirecording, sorting_agreement, folder=f'{base_comp_folder}\we')
                    except FileExistsError:
                        print('multiple comparaison waveform folder found, load from folder')
                        we = si.WaveformExtractor.load_from_folder(f'{base_comp_folder}\we', sorting=sorting_agreement)
                    if plot_comp:
                        print('Plot multiple comparaison summary in progress')
                        plot_maker(sorting_agreement, we, save, comp_multi_name, base_comp_folder, delay, mouse)
                        print('Plot multiple comparaison summary finished\n')

def quality_metrics(plot, save):
    for mouse in ['173']:
        print(mouse)
        #['Fixed_Delay', 'Random_Delay']
        for delay in ['Fixed_Delay']:
            
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

            sorter_name = 'herdingspikes_paramCustom'
            folder_path = fr'C:\local_data\Paper\Data\spike\{mouse}_{delay}'
            os.chdir(folder_path)
            sorter = ss.NpzSortingExtractor.load_from_folder(f'{sorter_name}\sorter\in_container_sorting')
            cs = CurationSorting(sorter)
            cs.merge([5, 11, 2, 22, 21, 16, 9])
            cs.merge([17, 18])
            cs.merge([14, 6])
            cs.merge([4, 0])
            cs.merge([10, 8])
            cs.merge([12, 15, 19])
            cs.merge([13, 20])
            clean_sorting = cs.sorting
            clean_sorting.save(folder=f'{sorter_name}\sorter_curated')
            we_curated = si.extract_waveforms(multirecording, clean_sorting , f'{sorter_name}\we_curated')
            if plot:
                plot_maker(clean_sorting, we_curated, plot, sorter_name, 'paramCustom', f'{folder_path}\{sorter_name}\we', delay, mouse)

""" global_job_kwargs = dict(n_jobs=-1, chunk_duration="1s")
si.set_global_job_kwargs(**global_job_kwargs) """
save = True
plot = True
run_multiple_sorter(save, nb_of_agreement=2, plot_sorter=False)

# quality_metrics(plot, save)
