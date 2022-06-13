# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:22:41 2022

@author: Gilles.DELBECQ
"""

import numpy as np
import matplotlib.pyplot as plt
import os, re

sampling_rate = 20000

selected_chan=[0,1,2,4,6]

def filter_signal(signal, order=3, sample_rate=sampling_rate, freq_low=300, freq_high=3000, axis=0):
    import scipy.signal
    Wn = [freq_low / (sample_rate / 2), freq_high / (sample_rate / 2)]
    sos_coeff = scipy.signal.iirfilter(order, Wn, btype="band", ftype="butter", output="sos")
    filtered_signal = scipy.signal.sosfiltfilt(sos_coeff, signal, axis=axis)
    return filtered_signal


folderpath = r'C:\Users\Gilles.DELBECQ\Desktop\Record tests\H5\15-02\raw'

list_dir = os.listdir(folderpath)
#    folderpath = folderpath
#    newpath = newpath

for file in list_dir:
    
    if file.endswith('.rbf'):

        print ('Filtering ' + file + '...')
        new_path = '%s/%s'%(folderpath,file)
        
    
    
        data = np.fromfile(new_path).reshape(16,-1)
        time_vector = np.arange(0,len(data[0])/sampling_rate,1/sampling_rate)
        filtered_signals =[]
        
        for i in range(len(data)):
            if i in selected_chan:
                signal_filtered = filter_signal(data[i])
                filtered_signals.append(signal_filtered)
        
        filtered_signals = np.array(filtered_signals)
        
        median = np.median(filtered_signals, axis=0)

        cmr_signals = filtered_signals-median     
        
        for i in range(len(filtered_signals)):
            signal = np.asarray(filtered_signals[i])
            plt.figure()
            plt.plot(time_vector,np.asarray(cmr_signals[i]))
            plt.plot(time_vector,signal)
            plt.title(rf'{file} Channel {selected_chan[i]}')
        
        
        
        name = re.sub('\.rbf$', '', file)
        
        file_save = '%s/%s_filtered.rbf'%(folderpath,name)
    
        with open(file_save, mode='wb') as file : 

                filtered_signals.tofile(file,sep='')                    
            
        print ('Conversion DONE')
        
        file_save = '%s/%s_filtered_cmr.rbf'%(folderpath,name)
        with open(file_save, mode='wb') as file : 

                filtered_signals.tofile(file,sep='')                    
            
        print ('Conversion DONE')
        
    else:
        print (file + ' is not an rbf file, will not be filtered')
        
print ('Whole directory has been converted successfully')





    
