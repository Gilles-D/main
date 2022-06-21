# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:15:39 2022

@author: gilles.DELBECQ
"""

import numpy as np
import matplotlib.pyplot as plt
import os


'''
Parameters
'''
sampling_rate = 20000
folderpath=r'E:\Data\Ephy\in vivo multiunit\In vivo Mars 2022\RBF\06-15'
Animal='2209'
Channels_to_plot=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
Save=True



files_to_analyse, record_lengths=[],[]

list_dir = os.listdir(folderpath)


for file in list_dir:
    if file.endswith('.rbf') and Animal in file and "cmr" in file:
        files_to_analyse.append('%s/%s'%(folderpath,file))


for index,file in np.ndenumerate(files_to_analyse):
    data = np.fromfile(file).reshape(16,-1)
    record_lengths.append(data.shape[1])
    
    if index[0] == 0:
        big_data = data
    else:
        big_data = np.concatenate((big_data,data),axis=1)


"""
Plot channels
"""   
fig, axs = plt.subplots(len(Channels_to_plot),sharex=True,sharey=True)
fig.suptitle(f'Animal : {Animal} Channels : {Channels_to_plot}')

time_vector = np.arange(0,len(big_data[0])/sampling_rate,1/sampling_rate)

for chan in range(len(Channels_to_plot)):
    axs[chan].plot(time_vector,big_data[chan,:])
    # axs[chan].get_yaxis().set_visible(False)
    
    for index,i in np.ndenumerate(record_lengths):
        if index[0] == 0:
            axs[chan].axvline(i/sampling_rate,c='red')
            old=i
        else:
            old=old+i
            axs[chan].axvline(old/sampling_rate,c='red')


"""
Save plot
"""  
if Save==True:
    fig.savefig(rf'{folderpath}\{Animal}_concatenated.png')
    file_save = rf'{folderpath}\{Animal}_concatenated.rbf'
    with open(file_save, mode='wb') as file : 
            big_data.tofile(file,sep='')                    

    