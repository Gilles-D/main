# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:57:54 2020

@author: Ludovic.SPAETH
"""

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

import numpy as np
import os 
import pandas as pd 
from matplotlib import pyplot as plt

import electroPyy.core.DetectPeaks as DP


#Where are the data ?
datafolder = 'U:/10_MOCAP/ThyOne_Beam_Analysis_from_corected_dataset'

#Give us that mouse name
animal = 'C1'

#Waht sessions should we compute today ?
sessions = ['baseline','stim','stim 2','stim 4','stim 5']

#Indicate the different markers to analyse and their color
markers = ['Back1','Back2','Back3','L_Foot2','R_Foot2']
colors = ['gray','skyblue','lightcoral','indianred','dodgerblue']

#Ref marker = marker on the back to cancel translational movemtn
ref_marker = 'Back1'

#Do we save anything ?
savefig = False



for session in sessions :
    
    if True:

        path = '{}/{}/{}'.format(datafolder,animal,session)
            
        file_list = ['{}/{}'.format(path,x) for x in os.listdir(path) if x.endswith('.xlsx') and 'Cal' not in x]
                
        #Group data -----------------------------------------------------------------------------
        #First the plot
        fig2,axx = plt.subplots(1,3,figsize=(14,4))
        plt.suptitle(path)
        axx[0].set_title('Y projection') ; axx[0].set_xlabel('X(mm)') ; axx[0].set_ylabel('Z(mm)')    
        axx[1].set_title('X projection') ; axx[1].set_xlabel('Y(mm)') ; axx[1].set_ylabel('Z(mm)')    
        axx[2].set_title('Z projection') ; axx[2].set_xlabel('X(mm)') ; axx[2].set_ylabel('Y(mm)')
        
        #Iterate for each marker in the subject
        for marker,color in zip(markers,colors):
            
            print(marker)
            
            #List to store the different 3D arrays, one for each trial
            STIM, NO_STIM = [],[]
            
            #Iterate over different trials (files)
            for file,idx in zip(file_list,range(len(file_list))):

                print (file.split('/')[-1])

                #Get ref marker positions                
                ref_marker_X = pd.read_excel(file,sheet_name=ref_marker,header=0,index_col=0).loc[:,'{}_X(mm)'.format(ref_marker)].values
                ref_marker_Y = pd.read_excel(file,sheet_name=ref_marker,header=0,index_col=0).loc[:,'{}_Y(mm)'.format(ref_marker)].values
                ref_marker_Z = pd.read_excel(file,sheet_name=ref_marker,header=0,index_col=0).loc[:,'{}_Z(mm)'.format(ref_marker)].values
                
                #Get all positions
                positions = pd.read_excel(file,sheet_name=marker,header=0,index_col=0)
                
                #Get XYZ coord
                X = positions.loc[:,'{}_X(mm)'.format(marker)].values
                Y = positions.loc[:,'{}_Y(mm)'.format(marker)].values
                Z = positions.loc[:,'{}_Z(mm)'.format(marker)].values
                
                #Substract ref coords to XYZ coord : cancels translational movement
                XX = X-ref_marker_X
                YY = Y-ref_marker_Y
                ZZ = Z
                
                #Stack normed YXZ coords in a single ndarray
                in_3D = np.vstack((XX,YY,ZZ))
                
                if 'no stim' in file: 
                    ls = '-'
                    NO_STIM.append(in_3D)
                    
                else:
                    ls = '--'
                    STIM.append(in_3D)
                
                #Plot all of this
                axx[0].plot(XX,Z,label=marker,color=color,alpha=0.5,linestyle=ls)
        
                axx[1].plot(YY,Z,label=marker,color=color,alpha=0.5,linestyle=ls)
                
                axx[2].plot(XX,Y,label=marker,color=color,alpha=0.5,linestyle=ls)
                
                
            #Change ndarray shape : concatenate the different trials 
            for i in range(len(NO_STIM)):
                if i == 0 :
                    no_stim_concatenated_coord = NO_STIM[i]
                else:
                    no_stim_concatenated_coord = np.concatenate((no_stim_concatenated_coord,NO_STIM[i]),axis=1)
                    
            for j in range(len(STIM)):
                if j == 0 :
                    stim_concatenated_coord = STIM[j]
                else:
                    stim_concatenated_coord = np.concatenate((stim_concatenated_coord,STIM[j]),axis=1)
                    
            #Detect peaks = detect each step episode based on its peaked amplitude
            
            if marker == 'L_Foot2' or marker == 'R_Foot2':
            
                plt.figure()
                plt.title('{} {} concatenated episodes (Z axis)'.format(session,marker))
                plt.plot(no_stim_concatenated_coord[2,:],label='No Stim')
                plt.plot(stim_concatenated_coord[2,:],label='Stim')
    
                #Get indexes for both conditions
                no_stim_indexes = DP.indexes(no_stim_concatenated_coord[2,:],mph=10,mpd=10,threshold=0,kpsh=True,valley=False)
                stim_indexes = DP.indexes(stim_concatenated_coord[2,:],mph=10,mpd=10,threshold=0,kpsh=True,valley=False)
                
                #Scatter the indexes to check
                plt.scatter(no_stim_indexes,no_stim_concatenated_coord[2,no_stim_indexes],label='No Stim Peaks')
                plt.scatter(stim_indexes,stim_concatenated_coord[2,stim_indexes],label='Stim Peaks')
                
                plt.legend(loc='best')
                
                #Functions to extract waveforms 
                def compute_waveform(signal, indexes, xspan=15):
                    return np.asarray([signal[x-xspan:x+xspan] for x in indexes])
                
                
                stim_waveforms=compute_waveform(stim_concatenated_coord[2,:],stim_indexes,xspan=15)
                no_stim_waveforms=compute_waveform(no_stim_concatenated_coord[2,:],no_stim_indexes,xspan=15)
                
                plt.figure()
                plt.title('{} {} Aligned Waveforms (Z axis)'.format(session,marker))
                for wave in stim_waveforms:
                    plt.plot(wave,color='skyblue',alpha=0.2)
                    
                for wave in no_stim_waveforms:
                    plt.plot(wave,color='gray',alpha=0.2)
                    
                plt.plot(np.nanmean(stim_waveforms,axis=0),color='blue',label='Stim Avg',linewidth=2)
                plt.plot(np.nanmean(no_stim_waveforms,axis=0),color='black',label='No Stim Avg',linewidth=2)
            

        if savefig == True:
            savepath = '{}/00_FIGURES'.format(path)
            
            try: 
                os.makedirs(savepath)
            except FileExistsError:
                pass
            
            plt.savefig('{}/00_recap_{}.pdf'.format(savepath,path.split('/')[-2]+path.split('/')[-1]))
            
