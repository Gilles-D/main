# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 15:28:47 2022

@author: Gilles.DELBECQ
"""

import numpy as np
import matplotlib.pyplot as plt
import os

'''
Parameters
'''
sampling_rate = 20000
sampling_rate_stim = 52.465


signal_file = r'//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/RBF/10-20/preprocessed/0004_06_01_0007_20000Hz_cmr.rbf' #use / at the end
number_of_channel=6

stim_idx_file=r'//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/stim_idx/10-20/0004_06_01_0007.txt'

Save=True


signal=np.fromfile(signal_file)
signal=signal.reshape(int(len(signal)/number_of_channel),-1).transpose()

stim_indexes=np.loadtxt(stim_idx_file)
stim_times=stim_indexes/sampling_rate_stim/sampling_rate

time_vector = np.arange(0,len(data[0])/sampling_rate,1/sampling_rate)

plt.plot(time_vector,signal[2])
for i in stim_times:
    plt.axvline(i)

for i in range(len(stim_times)):
    if i != 0:
        print(stim_times[i]-stim_times[i-1])

data_cmr=data_cmr.reshape(int(len(raw_file)/16),-1).transpose()

"""
Setups
"""
raw_path=rf'{folderpath}/raw/'
stim_idx_path=rf'{folderpath}/raw/'

files_to_analyse=[]

for file_name in os.listdir(raw_path):
    if file_name.endswith('.rbf') and Animal in file_name and not "filtered" in file_name and not "concatenated" in file_name:
        files_to_analyse.append(rf'{raw_path}/{file_name}')

if Save ==True:
    plot_save = rf"{folderpath}/plots"
    isExist = os.path.exists(plot_save)
    if not isExist:
        os.makedirs(plot_save) #Create folder for the experience if it is not already done



for file_to_analyze in files_to_analyse:
    
    raw_file = np.fromfile(file_to_analyze)
    
    name_file=file_to_analyze.split('/')[-1].split('.')[0]
    
    stim_idx_file=
    
    print(name_file)
    
    
    
    data = raw_file.reshape(int(len(raw_file)/16),-1).transpose()
        
    if Preprocessed == True:
        preprocessed_path=rf"{file_to_analyze.split('raw')[0]}/preprocessed"
        path_filter = rf"{preprocessed_path}/{name_file}_filtered.rbf"
        path_cmr = rf"{preprocessed_path}/{name_file}_cmr.rbf"
        
        data_filtered=np.fromfile(path_filter)
        data_filtered = data_filtered.reshape(int(len(raw_file)/16),-1).transpose()
        
        data_cmr=np.fromfile(path_cmr)
        data_cmr=data_cmr.reshape(int(len(raw_file)/16),-1).transpose()
    
    time_vector = np.arange(0,len(data[0])/sampling_rate,1/sampling_rate)
    
       
    
    """
    Plot every channel (raw, filtered, cmr) on individual plot
    """
    if Indivdual_plots==True:
        
        for i in range(len(data)):
            plt.figure()
            plt.title(rf'Channel {i}')
            # plt.plot(time_vector,data[i,:],alpha=1,linewidth=1)
            if Preprocessed == True:
                plt.plot(time_vector,data_filtered[i,:],alpha=0.5)
                plt.plot(time_vector,data_cmr[i,:])
                ax = plt.gca()
                ax.set_ylim(-0.0025,0.0025)
            
            if Save == True:
                plt.savefig(rf'{plot_save}/{name_file}_{i}.{plot_format}')
                
        if Autoclose == True:
            plt.close('all')
        
    """
    Plot all channel cmr on 1 plot
    """   
    if Preprocessed == True:
        fig1, axs = plt.subplots(len(data_cmr),sharex=True,sharey=True)
        fig1.suptitle(f'{file_to_analyze.split("/")[-1]} CMR of all channels')
        for i in range(len(data_cmr)):
            axs[i].plot(time_vector,data_cmr[i,:])
            axs[i].get_yaxis().set_visible(False)
        
        if Autoclose == True:
            plt.close()    
    
    """
    Plot all channel raw on 1 plot
    """
    fig2, axs = plt.subplots(len(data),sharex=True,sharey=True)
    fig2.suptitle(f'{file_to_analyze.split("/")[-1]} Raw signal of all channels')
    for i in range(len(data)):
        axs[i].plot(time_vector,data[i,:])
        axs[i].get_yaxis().set_visible(False)
    if Autoclose == True:
        plt.close()        
    """
    Save plots
    """
    if Save==True:
        if Preprocessed == True:
            fig1.savefig(rf'{plot_save}/{name_file}_cmr.{plot_format}')
        fig2.savefig(rf'{plot_save}/{name_file}_raw.{plot_format}')
