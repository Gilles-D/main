# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:55:27 2023

@author: Pierre.LE-CABEC
"""
from spikeinterface.postprocessing import (compute_spike_amplitudes, compute_unit_locations,
                                           compute_template_similarity, compute_correlograms,
                                           compute_noise_levels, compute_principal_components, compute_spike_locations)

import spikeinterface as si  # import core only
import spikeinterface.qualitymetrics as sqm
import spikeinterface_gui
import spikeinterface.widgets as sw
import spikeinterface.comparison as sc
from spikeinterface.exporters import export_to_phy
import os
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


folder_path = 'C:/_LMT/Data/spike_fede/173_Fixed_Delay_P0_first_ctrl'

# 'phy' 'soringview' 'spikeinterface_gui'
post_processing_visualisation = 'soringview'

############################
# Sorter outup comparaison #
comp_multi = sc.MultiSortingComparison.load_from_folder(f'{folder_path}/comp_multi')

sorting_agreement = comp_multi.get_agreement_sorting(minimum_agreement_count=2)

# # reload the waveform folder
we = si.WaveformExtractor.load_from_folder(f'{folder_path}/sorting_agrement_wf_folder', sorting=sorting_agreement)

if post_processing_visualisation == 'phy':
    
    if not os.path.isdir(f'{folder_path}/sorting_agrement_PHY'):
        # # some computations are done before to control all options
        compute_spike_amplitudes(we)
        compute_principal_components(we, n_components=3, mode='by_channel_global')
        
        # # the export process is fast because everything is pre-computed
        export_to_phy(we, output_folder=f'{folder_path}/sorting_agrement_PHY')
    
    print(f'Open Anaconda prompt and copy:\ncd {folder_path}/sorting_agrement_PHY\nphy template-gui params.py')
    

elif post_processing_visualisation == 'soringview':
    # # # some postprocessing is required
    noise = compute_noise_levels(we)
    amplitudes = compute_spike_amplitudes(we)
    unit_locations = compute_unit_locations(we)
    spike_locations = compute_spike_locations(we)
    similarity = compute_template_similarity(we)
    wecorrelograms, bins = compute_correlograms(we)
    
    
    qm_params = sqm.get_default_qm_params()
    
    qm = sqm.compute_quality_metrics(we, qm_params=qm_params)
    
    
    w1 = sw.plot_quality_metrics(we, display=False, backend="sortingview")
    w2 = sw.plot_sorting_summary(we, display=False, curation=True, backend="sortingview")
    
elif post_processing_visualisation == 'spikeinterface_gui':
    # This cerate a Qt app
    app = spikeinterface_gui.mkQApp() 
    # create the mainwindow and show
    win = spikeinterface_gui.MainWindow(we)
    win.show()
    # run the main Qt6 loop
    app.exec_()

else:
    raise ValueError(f'post_processing_visualisation not available: {post_processing_visualisation}')