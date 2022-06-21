# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 18:33:28 2019

@author: F.LARENO-FACCINI
"""

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp

path = r'D:/Working_Dir/In vivo Mars 2022/H5/06-15/2209_04_0004.h5'

f = h5.File(path,'r')
sigs = f['Data']['Recording_0']['AnalogStream']['Stream_0']['ChannelData']

tick = (f['Data']['Recording_0']['AnalogStream']['Stream_0']['InfoChannel'][0][9])/1000000 # in seconds
sampling_rate = 1/tick #in Hz


# fig, axs = plt.subplots(16)


# for i in range(len(sigs)):
#     signal = f['Data']['Recording_0']['AnalogStream']['Stream_0']['ChannelData'][i]
#     signal = np.asarray(signal)
#     signal_filtered = sp.savgol_filter(signal,51,3)
        
    
#     time = np.arange(0,len(signal)*tick,tick)
    
#     axs[i].plot(time,signal)
#     axs[i].plot(time,signal_filtered)
#     axs[i].set_ylim([-100,100])

for i in range(len(sigs)):
    signal = f['Data']['Recording_0']['AnalogStream']['Stream_0']['ChannelData'][i]
    signal = np.asarray(signal)
    signal_filtered = sp.savgol_filter(signal,51,3)
        
    
    time = np.arange(0,len(signal)/sampling_rate,1/sampling_rate)
    
    plt.figure()
    plt.plot(time,signal)
    plt.title('Channel %s'%(i))