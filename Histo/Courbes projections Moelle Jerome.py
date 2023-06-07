# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:51:32 2023

@author: Gilles.DELBECQ
"""

import pandas as pd
from matplotlib import pyplot as plt 

data_df = pd.read_excel('D:/Gilles.DELBECQ/Presentations/Poster Ecosse 2023/Figures/Heatmap moelle collat/Figure_Poster_Cervical.xlsx')

Slices_number = [1,2,3,4,5,6]

X_axis=list(data_df['DV Axis (µm)'])

plt.figure()

for i in Slices_number:
    Y_axis=data_df[rf'Slice {i}']
    plt.plot(Y_axis,X_axis, color='blue', alpha=0.1)
    
data_df['average'] = data_df[['Slice 1','Slice 2','Slice 3','Slice 4','Slice 5','Slice 6']].mean(axis=1)

plt.plot(data_df['average'],X_axis, color='black', alpha=1)
plt.title('Projections in Thoracic spinal cord')

plt.savefig(r'D:\Gilles.DELBECQ\Presentations\Poster Ecosse 2023\Figures\Heatmap moelle collat\Projections_Thoracic.svg')



