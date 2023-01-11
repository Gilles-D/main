# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:02:44 2019

@author: Isope
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 24 10:20:49 2012

@author: -
"""

###Permet de faire un cross correlograme entre deux 'neuron'.
###Si on veut un auto-corr, il suffit d entrer 2 fois le même numéro.
###Le fait de tout entrer manuellement ne permet pas d analyse de masse pour l instant
###Pas de sauvegarde implémentée pour l instant



from matplotlib import pyplot as plt
import scipy
import sys,os
import numpy as np
import pandas as pd


plotParamDefault =  {'plot_type' : {'value' : 'bar' , 'possible' : [ 'bar' , 'line']  },
                     'bin_width' : {'value' : 1.  , 'label' : 'bin width (ms)'},
                     'limit' : {'value' : 500.  , 'label' : 'max limit (ms)'},                            
                     'exact' : {'value' :False  , 'label' : 'Compute exact crosscorrelogram'},
                     'max_spike' : {'value' :20000  , 'label' : 'Otherwise bootstrap with'}}


width = plotParamDefault['bin_width']['value']/1000.
limit = plotParamDefault['limit']['value']/1000.


# =============================================================================
# LOAD SPIKETRAINS
# =============================================================================

# file_path=(r"C:\Users\Isope\Documents\DATA\FEDERICA\Group 10\5101\cluster2_5010-1600-P15_spike_times.txt")
# A = pd.read_csv(file_path, sep="\t",header=None) 
# A.columns = ['number', 'times']
# B=np.array(A['times'])
                             
# t1=B
# t2=B

# t1=np.array([0.02,0.04,0.06,0.08,0.12])
# t2=np.array([0.02,0.04,0.06,0.08,0.12])



"""
Load excel file with spike times and cluster #
"""
spike_times=pd.read_excel('//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/Analysis/PCA test/spike_times.xlsx', index_col=0).to_numpy().reshape(-1)

clusters=pd.read_excel('//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/Analysis/PCA test/test_waveform.xlsx', index_col=0).iloc[:, -1].to_numpy() 

"""
List clusters
"""
clusters_idx = np.unique(clusters)
clusters_idx = clusters_idx[~np.isnan(clusters_idx)]

spike_times = np.column_stack((spike_times,clusters))

t1=spikes_clustered =spike_times[spike_times[:, 1] == 0, :][:,0]
t2=spikes_clustered =spike_times[spike_times[:, 1] == 1, :][:,0]


# =============================================================================
# CALCULATE CROSSCORRELOGRAM
# =============================================================================

m1 = np.tile(t1[:,np.newaxis] , (1,t2.size) )
m2 = np.tile(t2[np.newaxis,:] , (t1.size,1) )
m = m2-m1
m = m.flatten()

y,x= np.histogram(m, bins = np.arange(-limit,limit, width), normed=False)
y[(int(limit / width)-1)] =0.

# shift predictor 
t3 = t1 + 0.0875
m1 = np.tile(t3[:,np.newaxis] , (1,t2.size) )
m2 = np.tile(t2[np.newaxis,:] , (t3.size,1) )
m = m2-m1
m = m.flatten()

y1, x1= np.histogram(m, bins = np.arange(-limit,limit, width), normed=False)

y1[(int(limit / width)-1)] =0.
y2=y-y1



# =============================================================================
# DISPLAY
# =============================================================================
fig = plt.figure()

x+=plotParamDefault['bin_width']['value']/1000
x1+=plotParamDefault['bin_width']['value']/1000

#première figure = cross correlograme
ax1 = fig.add_subplot(3,1,1)   
if plotParamDefault['plot_type']['value'] == 'bar':
    ax1.bar(x[:-1]*1000, y, width =width*1000)
elif plotParamDefault['plot_type']['value'] == 'line':
    ax1.plot(x[:-1]*1000, y )
ax1.set_xlim(-limit*1000., limit*1000.)
ax1.set_ylabel('CC', fontsize = 10) 

#deuxième figure = random

ax2 = fig.add_subplot(3,1,2)   
if plotParamDefault['plot_type']['value'] == 'bar':
    ax2.bar(x1[:-1]*1000, y1, width =width*1000)
elif plotParamDefault['plot_type']['value'] == 'line':
    ax2.plot(x1[:-1]*1000, y1 )
ax2.set_xlim(-limit*1000., limit*1000.)
ax2.set_ylabel('shift', fontsize = 10) 

#troisième figure = fig1 - fig2
ax3 = fig.add_subplot(3,1,3) 
if plotParamDefault['plot_type']['value'] == 'bar':
    ax3.bar(x1[:-1]*1000, y2, width =width*1000)
elif plotParamDefault['plot_type']['value'] == 'line':
    ax3.plot(x1[:-1]*1000, y2 )
ax3.set_xlim(-limit*1000., limit*1000.)
ax3.set_xlabel('Time (ms)', fontsize = 10)   
ax3.set_ylabel('shift - CC', fontsize = 10) 


