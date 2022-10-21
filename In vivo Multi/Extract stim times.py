# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 14:32:47 2022

@author: Gilles.DELBECQ
"""

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp
import os

experiments_to_analyse=['10-19']



folder_path = r"\\equipe2-nas1\Gilles.DELBECQ\Data\ePhy\Cohorte 1"

list_experiments = os.listdir(rf'{folder_path}/H5')

stim_idx_path=f'{folder_path}/stim_idx'
isExist = os.path.exists(stim_idx_path)
if not isExist:
    os.makedirs(stim_idx_path) #Create folder for the experience if it is not already done




for exp in experiments_to_analyse:
    h5_path=rf'{folder_path}/H5/{exp}/'
    
    for file in os.listdir(h5_path):
        if file.endswith('.h5'):
            print(file)
            f = h5.File(f'{h5_path}/{file}','r')
            try:
                events = f['Data']['Recording_0']['EventStream']['Stream_0']['EventEntity_0']
                stim_idx=events[0]
                
                name_file=file.split('.h5')[0]
                
                save_path= f'{stim_idx_path}\{exp}'
                isExist = os.path.exists(save_path)
                if not isExist:
                    os.makedirs(save_path) #Create folder for the experience if it is not already done
                    
                np.savetxt(f'{save_path}\{name_file}.txt',stim_idx)
            except:
                print(rf'{file} failed - no stim')