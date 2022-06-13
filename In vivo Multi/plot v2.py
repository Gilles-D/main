# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 16:19:38 2022

@author: MOCAP
"""


import numpy as np
import matplotlib.pyplot as plt

sampling_rate = 20000


path = r'D:/Working_Dir/In vivo Mars 2022/RBF/01-06/2209_012022-06-01T18-15-04_20000Hz.rbf'
file = np.fromfile(path).reshape(16,-1)

time_vector = np.arange(0,len(file[0])/sampling_rate,1/sampling_rate)

for i in range(len(file)):
    plt.figure()
    plt.title(rf'Channel {i}')
    plt.plot(time_vector,file[i,:])

# path = r'C:/Users/Gilles.DELBECQ/Desktop/Record tests/H5/14-02/1/2022-02-14T14-47-4614-02_20000Hz_filtered.rbf'
# file = np.fromfile(path).reshape(16,-1)

# time_vector = np.arange(0,len(file[0])/sampling_rate,1/sampling_rate)
# plt.plot(time_vector,file[3,:])