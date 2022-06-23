# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 16:19:38 2022

@author: MOCAP
"""


import numpy as np
import matplotlib.pyplot as plt
import os


'''
Parameters
'''
sampling_rate = 20000

path = r'D:/Working_Dir/In vivo Mars 2022/RBF/06-15/raw/2209_04_0006_20000Hz.rbf'

Save=False
Preprocessed=True
Indivdual_plots=True

"""
Setups
"""
raw_file = np.fromfile(path)
name_file=path.split('/')[-1].split('.')[0]
file = raw_file.reshape(int(len(raw_file)/16),-1).transpose()



if Preprocessed == True:
    preprocessed_path=rf"{path.split('raw')[0]}/preprocessed"
    
    path_filter = rf"{preprocessed_path}/{name_file}_filtered.rbf"
    path_cmr = rf"{preprocessed_path}/{name_file}_cmr.rbf"
    file_filtered=np.fromfile(path_filter).reshape(16,-1)
    file_cmr=np.fromfile(path_cmr).reshape(16,-1)

time_vector = np.arange(0,len(file[0])/sampling_rate,1/sampling_rate)



"""
Plot every channel (raw, filtered, cmr) on individual plot
"""
if Indivdual_plots==True:
    
    for i in range(len(file)):
        plt.figure()
        plt.title(rf'Channel {i}')
        # plt.plot(time_vector,file[i,:],alpha=1,linewidth=1)
        if Preprocessed == True:
            plt.plot(time_vector,file_filtered[i,:],alpha=0.5)
            plt.plot(time_vector,file_cmr[i,:])
    
    
"""
Plot all channel cmr on 1 plot
"""   
if Preprocessed == True:
    fig1, axs = plt.subplots(len(file_cmr),sharex=True,sharey=True)
    fig1.suptitle(f'{path.split("/")[-1]} CMR of all channels')
    for i in range(len(file_cmr)):
        axs[i].plot(time_vector,file_cmr[i,:])
        axs[i].get_yaxis().set_visible(False)


"""
Plot all channel raw on 1 plot
"""
fig2, axs = plt.subplots(len(file),sharex=True,sharey=True)
fig2.suptitle(f'{path.split("/")[-1]} Raw signal of all channels')
for i in range(len(file)):
    axs[i].plot(time_vector,file[i,:])
    axs[i].get_yaxis().set_visible(False)
    
"""
Save plots
"""
plot_save = rf"{path.split('raw')[0]}/plots"
isExist = os.path.exists(plot_save)
if not isExist:
    os.makedirs(plot_save) #Create folder for the experience if it is not already done
    
if Save==True:
    if Preprocessed == True:
        fig1.savefig(rf'{plot_save}/{name_file}_cmr.png')
    fig2.savefig(rf'{plot_save}/{name_file}_raw.png')
