# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 18:47:13 2022

@author: Gilles.DELBECQ
"""


import pandas as pd
import os
import numpy as np

folderpath=r'I:\Data\Microscopie\SOD\1222\Tif\mapping'

list_files=[]
for path, subdirs, files in os.walk(folderpath):
    for name in files:
        list_files.append(os.path.join(path, name))
        
        
all_data=pd.DataFrame()

start=0.7 # Medio-lat / AP position of 01 slice in cm
microscope_10x_scale=0.65*2 #µm/px


#Loop on files
i=0
for file in list_files:
    if file.endswith('.csv'):
        
        filename=file.split('Slice')
        print(filename[-1].split('.')[0])
        
        x = round(start-(i*0.1),2)
        csv_file=pd.read_csv(file)
        
        i=i+1
        all_data[x]=csv_file.iloc[:,[1]]
    
scaled = (np.array(range(len(all_data)))*microscope_10x_scale/1000).tolist() #/1000 to get it in mm
all_data['new_index']=scaled
all_data=all_data.set_index('new_index')
all_data.to_excel(rf"{folderpath}/data.xlsx")
