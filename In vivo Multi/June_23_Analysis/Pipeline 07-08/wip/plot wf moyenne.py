# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 01:45:22 2023

@author: Gil
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = 'D:/Seafile/Seafile/Ma bibliothèque/Rédaction/Manuscrits/Figures/Résultats/Data fig/waveforms/Unit_62_wf.xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# Drop the first column if it's just an index or irrelevant data
data = data.drop(data.columns[0], axis=1)

# Plotting signals from each of the 16 channels
plt.figure(figsize=(20, 15))
for i, column in enumerate(data.columns):
    plt.subplot(4, 4, i + 1)
    plt.plot(data[column])
    plt.title(f'Channel {i}')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
plt.savefig(rf"D:\Seafile\Seafile\Ma bibliothèque\Rédaction\Manuscrits\Figures\Résultats\Data fig\waveforms\Unit_62_wf.svg")
