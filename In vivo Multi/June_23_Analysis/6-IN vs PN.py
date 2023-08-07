import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks,peak_widths

import os
import pandas as pd

file_list =[
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_5 (Org_id_8_comp).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_8 (Org_id_14_moun).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_10 (Org_id_16_moun).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_14 (Org_id_20_moun).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_15 (Org_id_22_comp).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_16 (Org_id_12_spykingcircus).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_19 (Org_id_29_moun).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_20 (Org_id_30_comp).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_25 (Org_id_38_moun).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_29 (Org_id_48_moun).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_33 (Org_id_15_spykingcircus).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_29_07_allchan_allfiles/waveforms/Unit_45 (Org_id_6_spykingcircus).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_1 (Org_id_1_comp).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_4 (Org_id_5_comp).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_7 (Org_id_10_comp).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_11 (Org_id_15_moun).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_14 (Org_id_18_comp).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_18 (Org_id_23_comp).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_19 (Org_id_24_comp).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_21 (Org_id_28_moun).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_24 (Org_id_11_spykingcircus).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_25 (Org_id_24_spykingcircus).xlsx",
"D:/ePhy/SI_Data/spikesorting_results/0026_01_08_allchan_allfiles/waveforms/Unit_27 (Org_id_0_tdc).xlsx"
          ]



# Charger les données à partir des 50 fichiers CSV et stocker les waveforms dans une liste
waveforms = []
waveform_names = []

peak_properties=[]

for i,file in enumerate(file_list):
    file_path = file  # Remplacez par les noms de vos fichiers CSV
    df = pd.read_excel(file_path)
    waveform = df.values[:, :].T
    waveforms.append(waveform)
    waveform_names.append(os.path.basename(file))


for i,waveform in enumerate(waveforms):
    waveform_name=waveform_names[i]
    print(waveform_name)
    
    # Trouver l'indice du canal avec le maximum d'intensité
    channel_with_max_intensity = np.argmax(np.max(waveform, axis=1))
    
    # Extraire le waveform du canal avec le maximum d'intensité
    waveform_channel_max_intensity = waveform[channel_with_max_intensity]
    
    # Trouver les indices des pics négatifs
    negative_peak_indices = find_peaks(-waveform_channel_max_intensity,prominence=100)[0]
    # Visualiser le potentiel d'action et les mesures
    fig, ax = plt.subplots()
    time_points = np.arange(len(waveform_channel_max_intensity))
    ax.plot(time_points, waveform_channel_max_intensity, label=f'Canal {channel_with_max_intensity} (Pic)')

    
    # Il devrait y avoir un seul pic négatif, prenons son indice
    if len(negative_peak_indices) == 1:
        negative_peak_index = negative_peak_indices[0]
        
        width=peak_widths(-waveform_channel_max_intensity,negative_peak_indices,rel_height=0.5)
        
        ax.scatter(negative_peak_index, waveform_channel_max_intensity[negative_peak_index], color='r', label='Pic négatif')
        

        ax.hlines(-width[1],width[2],width[3], color="C2")
        
        # ax.axhline(half_height, color='g', linestyle='--', label='Demi-hauteur')
        # ax.axvline(half_height_index, color='g', linestyle='--')
        ax.set_xlabel('Temps (échantillons)')
        ax.set_ylabel('Potentiel (mV)')
        ax.legend()
        plt.show()
        
        properties = (waveform_name,waveform_channel_max_intensity[negative_peak_indices],width[0])
        
        peak_properties.append(properties)
        
    else:
        
        
        print("Aucun ou plusieurs pics négatifs détectés.")
        properties=(waveform_name,'Fail','Fail')
        peak_properties.append(properties)
        
    

peak_max = [t[1] for t in peak_properties]
half_width = [t[2] for t in peak_properties]

peak_max = [x for x in peak_max if not isinstance(x, str)]
half_width = [x for x in half_width if not isinstance(x, str)]

plt.scatter(peak_max,half_width)
