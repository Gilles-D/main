# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 18:33:28 2019

@author: F.LARENO-FACCINI
"""

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np

path = r'C:/Users/MOCAP/Documents/Multi Channel Systems/Multi Channel Experimenter/04-02/HDF5/2022-02-04T16-23-20Test_8489_Day2.h5'

f = h5.File(path,'r')
sigs = f['Data']['Recording_0']['AnalogStream']['Stream_0']['ChannelData'][5]
sigs = np.asarray(sigs)

plt.plot(sigs)
# list(f['Data']['Recording_0'])
# print(sigs)
# plt.plot(sigs)

info = f['Data']['Recording_0']['AnalogStream']['Stream_0']['InfoChannel']


# short_path = 'C:/Users/F.LARENO-FACCINI/Desktop/Trial/2019-10-03T15-30-24_1300um_P19_CrusI.rbf'
# long_path = 'C:/Users/F.LARENO-FACCINI/Desktop/Trial/2019-10-03T15-30-24_1300um_P19_CrusI_long.rbf'


# short = np.fromfile(short_path, dtype='float64').reshape(-1,16)
# long = np.fromfile(long_path, dtype='float64').reshape(-1,16)


# plt.plot(short[:,0], color='r')
# plt.plot(long[:,0], color='b', alpha=0.2)1000000 