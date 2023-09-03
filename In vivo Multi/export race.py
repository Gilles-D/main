# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 15:12:57 2023

@author: MOCAP
"""

import neo
from quantities import ms, s, Hz

# Chemin vers le fichier RHD
file_path = r'D:/ePhy/Intan_Data/5755/5755_08_08/5756_08_08_230808_173702/5756_08_08_230808_173702.rhd'

# Charger le fichier RHD
reader = neo.io.IntanIO(file_path)
block = reader.read_block()

reader.read_segment()


selected_channel = [1,6]
time_stop = 20*s

array_list = []

for chan in selected_channel:
    analog_signal = segment.analogsignals[0][:,chan]
    
    analog_signal = analog_signal.time_slice(t_start = 10*s,t_stop=time_stop)
    signal = np.array(analog_signal)
    array_list.append(signal)
    

final_array = np.column_stack((array_list[0],array_list[1]))

df = pd.DataFrame(final_array)

df.to_csv("D:\extracellular_signals_20kHz.csv")
