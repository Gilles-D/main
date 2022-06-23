# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:22:41 2022

@author: Gilles.DELBECQ
"""

import numpy as np
import matplotlib.pyplot as plt
import os, re


"""
PARAMETERS
"""

sampling_rate = 20000

selected_chan=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

freq_low = 300
freq_high = 3000
order = 2

folderpath = r'D:/Working_Dir/In vivo Mars 2022/RBF/' #use /


def filter_signal(signal, order=order, sample_rate=sampling_rate, freq_low=freq_low, freq_high=freq_high, axis=0):
    import scipy.signal
    Wn = [freq_low / (sample_rate / 2), freq_high / (sample_rate / 2)]
    sos_coeff = scipy.signal.iirfilter(order, Wn, btype="band", ftype="butter", output="sos")
    filtered_signal = scipy.signal.sosfiltfilt(sos_coeff, signal, axis=axis)
    return filtered_signal

list_files=[]
for path, subdirs, files in os.walk(folderpath):
    for name in files:
        list_files.append(os.path.join(path, name))

for file in list_files:
    
    if file.endswith('.rbf') and not "filtered" in file and not "concatenated" in file:
        name = file.split('\\')[-1].split('.rbf')[0]
        path = file.split('\\')[0]
        
        print ('Filtering ' + file + '...')
        # new_path = rf'{folderpath}{file}'
      
        
        # data = np.fromfile(new_path).reshape(16,-1)
        
        raw_file = np.fromfile(file)
        data = raw_file.reshape(int(len(raw_file)/16),-1).transpose()
        
        time_vector = np.arange(0,len(data[0])/sampling_rate,1/sampling_rate)
        
        
        filtered_signals =[]
              
        
        for i in range(len(data)):
            if i in selected_chan:
                signal_filtered = filter_signal(data[i])
                filtered_signals.append(signal_filtered)
        
        filtered_signals = np.array(filtered_signals)
        
        median = np.median(filtered_signals, axis=0)

        cmr_signals = filtered_signals-median     

        
        save_path = rf'{path}/preprocessed/'
        
        isExist = os.path.exists(save_path)
        if not isExist:
            os.makedirs(save_path) #Create folder for the experience if it is not already done
        
        
        file_save=rf'{save_path}/{name}_filtered.rbf'
        with open(file_save, mode='wb') as file : 

                filtered_signals.tofile(file,sep='')                    
            
                print ('Filter DONE')
        
        file_save=rf'{save_path}/{name}_cmr.rbf'
        with open(file_save, mode='wb') as file : 

                cmr_signals.tofile(file,sep='')                    
            
                print ('CMR DONE')
        
    else:
        print (file + ' is not an rbf file, will not be filtered')
        
print ('Whole directory has been converted successfully')