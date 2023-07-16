# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:03:36 2023

@author: _LMT
"""
import os
import spikeinterface as si  # import core only
import spikeinterface.preprocessing as spre
import probeinterface as pi
import pandas as pd
import pickle

def concatenate_ephy_data():
    mouse_dict = {
                    '173': {'group': '15', 'gender': 'Female', 'delay': {'Random Delay': [
                                                                                            [{'P0': ['No Stim']},],
                                                                                            [{'P13': ['No Stim', 'Stim']},],
                                                                                            [{'P15': ['No Stim', 'Stim']},],
                                                                                            [{'P16': ['No Stim', 'Stim']},],
                                                                                            [{'P18': ['No Stim', 'Stim']}],
                                                                                        ],
                                                                        'Fixed Delay': [
                                                                                            [{'P0': ['No Stim']},],
                                                                                            [{'P13': ['No Stim', 'Stim']},],
                                                                                            [{'P15': ['No Stim', 'Stim']},],
                                                                                            [{'P16': ['No Stim', 'Stim']},],
                                                                                            [{'P18': ['No Stim', 'Stim']}],
                                                                                        ],
                                                                        },
                            },
                    '174': {'group': '15', 'gender': 'Female', 'delay': {'Random Delay': [
                                                                                            [{'P0': ['No Stim']},],
                                                                                            [{'P13': ['No Stim', 'Stim']},],
                                                                                            [{'P15': ['No Stim', 'Stim']},],
                                                                                            [{'P16': ['No Stim', 'Stim']},],
                                                                                            [{'P18': ['No Stim', 'Stim']}],
                                                                                        ],
                                                                        'Fixed Delay': [
                                                                                            [{'P0': ['No Stim']},],
                                                                                            [{'P13': ['No Stim', 'Stim']},],
                                                                                            [{'P15': ['No Stim', 'Stim']},],
                                                                                            [{'P16': ['No Stim', 'Stim']},],
                                                                                            [{'P18': ['No Stim', 'Stim']}],
                                                                                        ],
                                                                        }
                            },
                    '176': {'group': '15', 'gender': 'Female', 'delay': {'Random Delay': [
                                                                                            [{'P0': ['No Stim']},],
                                                                                            [{'P13': ['No Stim', 'Stim']},],
                                                                                            [{'P15': ['No Stim', 'Stim']},],
                                                                                            [{'P16': ['No Stim', 'Stim']},],
                                                                                            [{'P18': ['No Stim', 'Stim']}],
                                                                                        ],
                                                                        'Fixed Delay': [
                                                                                            [{'P0': ['No Stim']},],
                                                                                            [{'P13': ['No Stim', 'Stim']},],
                                                                                            [{'P15': ['No Stim', 'Stim']},],
                                                                                            [{'P16': ['No Stim', 'Stim']},],
                                                                                            [{'P18': ['No Stim', 'Stim']}],
                                                                                        ],
                                                                        }
                            },

                    '6401': {'group': '14', 'gender': 'Male', 'delay': {'Random Delay': [
                                                                                                [{'P13': ['No Stim', 'Stim']},],
                                                                                            [{'P15': ['No Stim', 'Stim']},],
                                                                                            [{'P16': ['No Stim', 'Stim']},],
                                                                                            [{'P18': ['No Stim', 'Stim']}],
                                                                                        ],
                                                                        'Fixed Delay': [
                                                                                            [{'P13': ['No Stim', 'Stim']},],
                                                                                            [{'P15': ['No Stim', 'Stim']},],
                                                                                            [{'P16': ['No Stim', 'Stim']},],
                                                                                            [{'P18': ['No Stim', 'Stim']}],
                                                                                        ],
                                                                        }
                            },
                    '6402': {'group': '14', 'gender': 'Male', 'delay': {'Random Delay': [
                                                                                            [{'P13': ['No Stim', 'Stim']},],
                                                                                            [{'P15': ['No Stim', 'Stim']},],
                                                                                            [{'P16': ['No Stim', 'Stim']},],
                                                                                            [{'P18': ['No Stim', 'Stim']}],
                                                                                        ],
                                                                        'Fixed Delay': [
                                                                                            [{'P13': ['No Stim', 'Stim']},],
                                                                                            [{'P15': ['No Stim', 'Stim']},],
                                                                                            [{'P16': ['No Stim', 'Stim']},],
                                                                                            [{'P18': ['No Stim', 'Stim']}],
                                                                                        ],
                                                                            },
                            },
                    '6409': {'group': '14', 'gender': 'Female', 'delay': {  'Random Delay': [
                                                                                                [{'P13': ['No Stim', 'Stim']},],
                                                                                                [{'P15': ['No Stim', 'Stim']},],
                                                                                                [{'P16': ['No Stim', 'Stim']},],
                                                                                                [{'P18': ['No Stim', 'Stim']}],
                                                                                            ],
                                                                            'Fixed Delay': [
                                                                                                [{'P13': ['No Stim', 'Stim']},],
                                                                                                [{'P15': ['No Stim', 'Stim']},],
                                                                                                [{'P16': ['No Stim', 'Stim']},],
                                                                                                [{'P18': ['No Stim', 'Stim']}],
                                                                                            ],
                                                                        }
                            },
                    '6457': {'group': '17', 'gender': 'Female', 'delay': {  'Random Delay': [
                                                                                                [{'P0': ['No Stim']},],
                                                                                                [{'P15': ['No Stim', 'Stim']},],
                                                                                                [{'P16': ['No Stim', 'Stim']},],
                                                                                                [{'P18': ['No Stim', 'Stim']},],
                                                                                                [{'P13': ['No Stim', 'Stim']}],
                                                                                            ],
                                                                            'Fixed Delay': [
                                                                                                [{'P0': ['No Stim']},],
                                                                                                [{'P16': ['No Stim', 'Stim']},],
                                                                                                [{'P13': ['No Stim', 'Stim']},],
                                                                                                [{'P15': ['No Stim', 'Stim']},],
                                                                                                [{'P18': ['No Stim', 'Stim']}],
                                                                                            ],
                                                                        }
                            },
                    '6456': {'group': '17', 'gender': 'Female', 'delay': {  'Random Delay': [
                                                                                                [{'P0': ['No Stim']},],
                                                                                                [{'P13': ['No Stim', 'Stim']},],
                                                                                                [{'P18': ['No Stim', 'Stim']},],
                                                                                                [{'P15': ['No Stim', 'Stim']},],
                                                                                                [{'P16': ['No Stim', 'Stim']}],
                                                                                            ],
                                                                            'Fixed Delay': [
                                                                                                [{'P0': ['No Stim']},],
                                                                                                [{'P18': ['No Stim', 'Stim']},],
                                                                                                [{'P13': ['No Stim', 'Stim']},],
                                                                                                [{'P16': ['No Stim', 'Stim']},],
                                                                                                [{'P15': ['No Stim', 'Stim']}],
                                                                                            ],
                                                                        }
                            },
                    '6924': {'group': '16', 'gender': 'Male', 'delay': {'Random Delay': [
                                                                                            [{'P0': ['No Stim']},],
                                                                                            [{'P13': ['No Stim', 'Stim']},],
                                                                                            [{'P15': ['No Stim', 'Stim']},],
                                                                                            [{'P16': ['No Stim', 'Stim']},],
                                                                                            [{'P18': ['No Stim', 'Stim']}],
                                                                                        ],
                                                                        'Fixed Delay': [
                                                                                            [{'P0': ['No Stim']},],
                                                                                            [{'P15': ['No Stim', 'Stim']},],
                                                                                            [{'P13': ['No Stim', 'Stim']},],
                                                                                            [{'P18': ['No Stim', 'Stim']},],
                                                                                            [{'P16': ['No Stim', 'Stim']}],
                                                                                        ],
                                                                        }
                            },
                    '6928': {'group': '16', 'gender': 'Male', 'delay': { 'Random Delay': [
                                                                                            [{'P0': ['No Stim']},],
                                                                                            [{'P13': ['No Stim', 'Stim']},],
                                                                                            [{'P18': ['No Stim', 'Stim']},],
                                                                                            [{'P15': ['No Stim', 'Stim']},],
                                                                                            [{'P16': ['No Stim', 'Stim']}],
                                                                                        ],
                                                                        'Fixed Delay': [
                                                                                            [{'P0': ['No Stim']},],
                                                                                            [{'P18': ['No Stim', 'Stim']},],
                                                                                            [{'P16': ['No Stim', 'Stim']},],
                                                                                            [{'P13': ['No Stim', 'Stim']},],
                                                                                            [{'P15': ['No Stim', 'Stim']}],
                                                                                        ],
                                                                        }
                            },
                }
    for mouse_id in mouse_dict.keys():
        print(mouse_id)
        group = mouse_dict[mouse_id]['group']
        gender = mouse_dict[mouse_id]['gender']
        if mouse_id != '174':
            continue
        for delay in mouse_dict[mouse_id]['delay']:
            print(delay)
            trial_time_index_df = pd.DataFrame()
            list_recording = []
            for session_order_nb, session_list in enumerate(mouse_dict[mouse_id]['delay'][delay]):
                if group == '14':
                    session_order_nb += 1
                for session_dict in session_list:
                    for protocol, condition_list in session_dict.items():
                        for condition in condition_list:
                            print(session_order_nb, protocol,condition)
                            current_ephy_folder_path = (fr'\\Equipe2-nas1\f.lareno-faccini\BACKUP FEDE\Ephy\Group {group}\{mouse_id} (CM16-Buz - {gender})\{delay}\{protocol}\{condition}')
                            ###########################################
                            # raw recording extraction and processing #
                            trial_nb = 0
                            file_list = [f'{current_ephy_folder_path}\{file}' for file in os.listdir(f'{current_ephy_folder_path}') if file.split('.')[-1] == 'rbf']
                            file_list.sort()
                            for file in file_list:
                                print(os.path.basename(file))
                                trial_nb += 1
                                if 'Empty' in file: #TODO implemant a way to track back what part of the concatenated recording correspond to what trial (especialy because of Empty file)
                                    print('Empty recording')
                                    continue
                                else:
                                        list_recording.append(si.read_binary(file,
                                                                            num_chan=16, 
                                                                            sampling_frequency=20000,
                                                                            dtype='float64', 
                                                                            gain_to_uV=0.000001, 
                                                                            offset_to_uV=0, 
                                                                            time_axis=0))
                                        trial_time_index_df = pd.concat((trial_time_index_df, pd.DataFrame({'protocol': [protocol]* len(list_recording[-1].get_times()),
                                                                                                            'protocol_order': session_order_nb,
                                                                                                            'condition': [condition]* len(list_recording[-1].get_times()),
                                                                                                            'trial_nb': [trial_nb]* len(list_recording[-1].get_times()),
                                                                                                            'time': list_recording[-1].get_times(),})))

            multirecording = si.concatenate_recordings(list_recording)
            trial_time_index_df = trial_time_index_df.reset_index()
            del trial_time_index_df['index']
            trial_time_index_df['concatenated_time'] = multirecording.get_times()

            # with open(f'C:\local_data\Paper\Data\concaneted_recording\{mouse_id}_{delay}\concatenated_recording_trial_time_index_df.pickle', 'wb') as file:
            #     pickle.dump(trial_time_index_df, file, protocol=pickle.HIGHEST_PROTOCOL)
            # del list_recording

            probe = pi.io.read_probeinterface('C:\local_data\Paper\Data\spike\CM16_Buz_Sparse/CM16_Buz_Sparse.json')
            probe = probe.probes[0]
            multirecording = multirecording.set_probe(probe)
            multirecording = spre.bandpass_filter(multirecording, freq_min=300, freq_max=6000)
            multirecording = spre.common_reference(multirecording, reference='global', operator='median')
            multirecording.annotate(is_filtered=True)
            # multirecording = multirecording.save(folder=f"C:\local_data\Paper\Data\concaneted_recording\{mouse_id}_{delay}\concatenated_recording")
            del multirecording

if __name__ == "__main__":
    concatenate_ephy_data()