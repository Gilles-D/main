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

# path = r'D:/Working_Dir/In vivo Mars 2022/RBF/06-15/raw/2209_04_0006_20000Hz.rbf'

folderpath = r'\\equipe2-nas1\Gilles.DELBECQ\Data\ePhy\Anesthesie\RBF\4713/' #use / at the end
Animal='4713'

Save=False
plot_format='png'

Preprocessed=True
Indivdual_plots=False

Autoclose=False



"""
Setups
"""
raw_path=rf'{folderpath}/raw/'
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
