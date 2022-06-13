# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 16:19:38 2022

@author: MOCAP
"""


import numpy as np
import matplotlib.pyplot as plt

sampling_rate = 20000


path = r'C:/Users/Gilles.DELBECQ/Desktop/Record tests/H5/15-02/2022-02-15T17-09-0315-02_20000Hz.rbf'
file = np.fromfile(path).reshape(16,-1)

time_vector = np.arange(0,len(file[0])/sampling_rate,1/sampling_rate)

plt.figure()
plt.plot(time_vector,file[0,:])

# path = r'C:/Users/Gilles.DELBECQ/Desktop/Record tests/H5/14-02/1/2022-02-14T14-47-4614-02_20000Hz_filtered.rbf'
# file = np.fromfile(path).reshape(16,-1)

# time_vector = np.arange(0,len(file[0])/sampling_rate,1/sampling_rate)
# plt.plot(time_vector,file[3,:])