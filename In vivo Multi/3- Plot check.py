# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 16:19:38 2022

@author: MOCAP
"""


import numpy as np
import matplotlib.pyplot as plt


'''
Parameters
'''
sampling_rate = 20000

path = r'E:/Data/Ephy/in vivo multiunit/In vivo Mars 2022/RBF/06-15/2989_03_0001_20000Hz.rbf'
path_filter = rf'{path.split(".")[0]}_filtered.rbf'
path_cmr = rf'{path.split(".")[0]}_filtered_cmr.rbf'



file = np.fromfile(path).reshape(16,-1)
file_filtered=np.fromfile(path_filter).reshape(16,-1)
file_cmr=np.fromfile(path_cmr).reshape(16,-1)

time_vector = np.arange(0,len(file[0])/sampling_rate,1/sampling_rate)

Save=False

"""
Plot every channel (raw, filtered, cmr) on individual plot
"""
#     for i in range(len(file)):
#     plt.figure()
#     plt.title(rf'Channel {i}')
#     plt.plot(time_vector,file[i,:],alpha=0.25)
#     plt.plot(time_vector,file_filtered[i,:],alpha=0.5)
#     plt.plot(time_vector,file_cmr[i,:])
    
    
"""
Plot all channel cmr on 1 plot
"""   
fig1, axs = plt.subplots(len(file_cmr),sharex=True,sharey=True)
fig1.suptitle(f'{path.split("/")[-1]} CMR of all channels')
for i in range(len(file_cmr)):
    axs[i].plot(time_vector,file_cmr[i,:])
    # axs[i].get_yaxis().set_visible(False)


"""
Plot all channel raw on 1 plot
"""
fig2, axs = plt.subplots(len(file),sharex=True,sharey=True)
fig2.suptitle(f'{path.split("/")[-1]} Raw signal of all channels')
for i in range(len(file)):
    axs[i].plot(time_vector,file[i,:])
    # axs[i].get_yaxis().set_visible(False)
    
"""
Save plots
"""
if Save==True:
    fig1.savefig(rf'{path.split(".")[0]}_cmr.png')
    fig2.savefig(rf'{path.split(".")[0]}_raw.png')
