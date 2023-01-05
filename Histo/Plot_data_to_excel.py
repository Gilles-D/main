# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 18:47:13 2022

@author: Gilles.DELBECQ
"""


import pandas as pd
import os

folderpath=r'\\equipe2-nas1\Gilles.DELBECQ\Data\Microscopie\Histo électrodes\Injections Chr2 retro\0001\plot_analysis'

list_files=[]
for path, subdirs, files in os.walk(folderpath):
    for name in files:
        list_files.append(os.path.join(path, name))
        
        
all_data=pd.DataFrame()

start=1.92 # Medio-lat position of 01 slice in cm

#Loop on files
i=0
for file in list_files:
    if file.endswith('.csv'):
        x = round(start-(i*0.1),2)
        csv_file=pd.read_csv(file)
        
        i=i+1
        all_data[x]=csv_file.iloc[:,[1]]
    
all_data.to_excel(rf"{folderpath}/data.xlsx")