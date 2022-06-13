# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:22:22 2019

@author: gille

Extraire le temps du premier spike après chaque stim, s'il est dans la fenêtre des 20 ms'

"""
import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt 
manip = '090519-11'
file_path = 'D:/Analyses\Fig 8\Figure\Premier spike/{}.xlsx'.format(manip)
file = 'D:/Analyses/Fig 8/Figure/Premier spike/{}_spike_times.xlsx'.format(manip)

# Prendre les temps de stim
stim_idx = pd.read_excel(file_path)
stim_idx = stim_idx.values.T[0].tolist()
spike_idx = pd.read_excel(file)
spike_idx = spike_idx.values.T[0].tolist()
fenetre = 20/1000
n=0
j=0
k=0

RASTER = []
N = []

for idx in stim_idx:
    # print(idx)
    n = n + 1
    for s_idx in spike_idx:
        # print(s_idx)
        if k < n:
            if s_idx > idx and s_idx <= idx+fenetre:
                print(s_idx)
                print('est compris entre')
                print(idx)
                print('et')
                print(idx+fenetre)
                spike_raster = s_idx - idx
                # print(n)
                N.append([n])
                RASTER.append([spike_raster*1000])
                k = k + 1
            else:
                N.append([n])
                
            


# # print(RASTER)
plt.figure()
plt.eventplot(RASTER, lineoffsets = 3, linelengths = 3, colors = 'black')
# plt.scatter(np.average(RASTER))
plt.title('Peri Stimulus Histogramme')
plt.xlabel('Temps en ms')
plt.ylabel("Nombre d'événements")
plt.axvspan(0, 5, alpha=0.1, color='blue')
plt.xlim(-19,22)
plt.ylim(-9,115)
# plt.savefig("D:/Analyses/Fig E.1/Figure/RASTER32.svg", transparent=True)


# bins = 200
# plt.figure()
# plt.hist(np.array(RASTER), bins=bins, color='black')
# plt.title("Peri Stimulus Histogramme. Bins = {}".format(bins))
# plt.axvspan(0, 5, alpha=0.1, color='blue')
# plt.xlim(-19,22)
# plt.ylim(0,105)
# plt.show()
# plt.savefig("D:/Analyses/Fig E.1/Figure/PSTH32.svg", transparent=True)


# df5 = pd.DataFrame(RASTER)
# df5.to_excel('D:/Analyses/Fig E.1/Figure/Premier spike/data/{}.xlsx'.format(manip))