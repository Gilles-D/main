# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 18:40:24 2020

Extracts xcl files containing marker position for each trial of a session + builds raw plot for trajectories

@author: Ludovic.SPAETH
"""

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
import numpy as np
import os 
import pandas as pd 
from matplotlib import pyplot as plt

#path = 'U:/10_MOCAP/ThyOne_Beam_Analysis_from_corected_dataset/C1/stim'

sessions = ['baseline','stim','stim 2','stim 4','stim 5']



animal = 'D2'


for session in sessions :
    
    try:

        path = 'U:/10_MOCAP/ThyOne_Beam_Analysis_from_corected_dataset/{}/{}'.format(animal,session)
            
        file_list = ['{}/{}'.format(path,x) for x in os.listdir(path) if x.endswith('.xlsx') and 'Cal' not in x]
        
        markers = ['Back1','Back2','Back3','L_Foot2','R_Foot2']
        colors = ['gray','skyblue','lightcoral','indianred','dodgerblue']
        
        ref_marker = 'Back1'
        
        savefig = True
        
        #Plot each file ---------------------------------------------------------------
        for file in file_list:
            
        
            fig,ax = plt.subplots(1,2,figsize=(14,4))
            
            ref_marker_X = pd.read_excel(file,sheet_name=ref_marker,header=0,index_col=0).loc[:,'{}_X(mm)'.format(ref_marker)].values
            ref_marker_Y = pd.read_excel(file,sheet_name=ref_marker,header=0,index_col=0).loc[:,'{}_Y(mm)'.format(ref_marker)].values
            ref_marker_Z = pd.read_excel(file,sheet_name=ref_marker,header=0,index_col=0).loc[:,'{}_Z(mm)'.format(ref_marker)].values
        
            
            for marker,color in zip(markers,colors) : 
                
                positions = pd.read_excel(file,sheet_name=marker,header=0,index_col=0)
                
                X = positions.loc[:,'{}_X(mm)'.format(marker)].values
                Y = positions.loc[:,'{}_Y(mm)'.format(marker)].values
                Z = positions.loc[:,'{}_Z(mm)'.format(marker)].values
                
                ax[0].plot(Y,Z,label=marker,color=color)
                
                if marker == ref_marker:
                    continue
                else: 
                    ax[1].plot(Y-ref_marker_Y,Z,label=marker)
                    
            ax[0].set_title(file.split('/')[-1])
            ax[0].legend(loc='best')
            ax[0].set_xlabel('Y(mm)') ; ax[0].set_ylabel('Z(mm)')
            ax[0].set_xlim(0,500)
            
            ax[1].set_title('Normed Trajectories')
            ax[1].legend(loc='best')
            ax[1].set_xlabel('Y(mm)') ; ax[1].set_ylabel('Z(mm)')
            
            if savefig == True:
                savepath = '{}/00_FIGURES'.format(path)
                
                try: 
                    os.makedirs(savepath)
                except FileExistsError:
                    pass
                
                plt.savefig('{}/{}.pdf'.format(savepath,file.split('/')[-1][:-5]))
                
                
        
        
        #Group data -------------------------------------------------------------------
        fig2,axx = plt.subplots(1,3,figsize=(14,4))
        plt.suptitle(path)
        axx[0].set_title('Y projection') ; axx[0].set_xlabel('X(mm)') ; axx[0].set_ylabel('Z(mm)')
        
        axx[1].set_title('X projection') ; axx[1].set_xlabel('Y(mm)') ; axx[1].set_ylabel('Z(mm)')
        
        axx[2].set_title('Z projection') ; axx[2].set_xlabel('X(mm)') ; axx[2].set_ylabel('Y(mm)')
        
        
        
        left_foot, right_foot,left_hip,right_hip = [],[],[],[]
            
        for marker,color in zip(markers,colors):
            
            XX, YY, ZZ = [],[],[]
            
            for file,idx in zip(file_list,range(len(file_list))):
                
                if 'no stim' in file: 
                    ls = '-'
                else:
                    ls = '--'
                    
                ref_marker_X = pd.read_excel(file,sheet_name=ref_marker,header=0,index_col=0).loc[:,'{}_X(mm)'.format(ref_marker)].values
                ref_marker_Y = pd.read_excel(file,sheet_name=ref_marker,header=0,index_col=0).loc[:,'{}_Y(mm)'.format(ref_marker)].values
                ref_marker_Z = pd.read_excel(file,sheet_name=ref_marker,header=0,index_col=0).loc[:,'{}_Z(mm)'.format(ref_marker)].values
                
                positions = pd.read_excel(file,sheet_name=marker,header=0,index_col=0)
                
                X = positions.loc[:,'{}_X(mm)'.format(marker)].values
                Y = positions.loc[:,'{}_Y(mm)'.format(marker)].values
                Z = positions.loc[:,'{}_Z(mm)'.format(marker)].values
                
                axx[0].plot(X-ref_marker_X,Z,label=marker,color=color,alpha=0.5,linestyle=ls)
        
                axx[1].plot(Y-ref_marker_Y,Z,label=marker,color=color,alpha=0.5,linestyle=ls)
                
                axx[2].plot(X-ref_marker_X,Y,label=marker,color=color,alpha=0.5,linestyle=ls)
                
                if idx == 0:
                    XX = X-ref_marker_X
                    YY = Y-ref_marker_Y
                    ZZ = Z-ref_marker_Z
                    
                else:
                    XX = np.concatenate((XX,X-ref_marker_X))
                    YY = np.concatenate((YY,Y-ref_marker_Y))
                    ZZ = np.concatenate((ZZ,Z-ref_marker_Z))
                    
        
        
          
        if savefig == True:
            savepath = '{}/00_FIGURES'.format(path)
            
            try: 
                os.makedirs(savepath)
            except FileExistsError:
                pass
            
            plt.savefig('{}/00_recap_{}.pdf'.format(savepath,path.split('/')[-2]+path.split('/')[-1]))
                
        #        XX.append(np.asarray(X-ref_marker_X))
        #        YY.append(np.asarray(Y-ref_marker_Y))
        #        ZZ.append(np.asarray(Z))
                
    except :
        
        print('Session {} bugued somewhere'.format(session))
                
        
        continue