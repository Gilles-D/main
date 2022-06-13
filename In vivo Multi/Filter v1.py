# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:22:41 2022

@author: Gilles.DELBECQ
"""

import numpy as np
import matplotlib.pyplot as plt

sampling_rate = 20000

def filter_signal(signal, order=3, sample_rate=sampling_rate, freq_low=300, freq_high=3000, axis=0):
    import scipy.signal
    Wn = [freq_low / (sample_rate / 2), freq_high / (sample_rate / 2)]
    sos_coeff = scipy.signal.iirfilter(order, Wn, btype="band", ftype="butter", output="sos")
    filtered_signal = scipy.signal.sosfiltfilt(sos_coeff, signal, axis=axis)
    return filtered_signal



path = r'C:/Users/Gilles.DELBECQ/Desktop/Record tests/H5/07-02/2022-02-07T15-32-15Test_8489_Day3_20000Hz.rbf'
file = np.fromfile(path).reshape(16,-1)

time_vector = np.arange(0,len(file[0])/sampling_rate,1/sampling_rate)

filtered_signals =[]


i=0
for signal in file:
    signal_filtered = filter_signal(signal)
    filtered_signals.append(signal_filtered)
    # plt.figure()
    # plt.plot(time_vector,signal_filtered, color='red')
    # plt.plot(time_vector,signal, color='grey', alpha=0.5)
    # plt.title("Channel %s"%(i))
    i=+1
    
file_save = r'C:/Users/Gilles.DELBECQ/Desktop/Record tests/H5/07-02/2022-02-07T15-32-15Test_8489_Day3_20000Hz_filtered.rbf'

a = np.array(filtered_signals)

with open(file_save, mode='wb') as file : 

        a.tofile(file,sep='')  