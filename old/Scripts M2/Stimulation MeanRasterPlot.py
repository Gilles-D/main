# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:22:22 2019

@author: gille

"""
import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt 
import os
from scipy import stats 
n = 0
N = []
Datapath1 = 'D:/Analyses/Fig 8/Figure/Premier spike/data'
RASTER =[]
List_File_paths1 = []
SEM = []
for r, d, f in os.walk(Datapath1):
# r=root, d=directories, f = files
    for filename in f:
        if '.xls' in filename:
            List_File_paths1.append(os.path.join(r, filename))

           
for file_path in List_File_paths1:    
    Moyenne = np.average(np.array(pd.read_excel(file_path, usecols='B'))) #ms
    sem = stats.sem(np.array(pd.read_excel(file_path, usecols='B')))
    SEM.append(sem)
    RASTER.append(Moyenne)
    N.append(n)
    n = n + 1
    plt.scatter(Moyenne,n,s=1, color='black')
    plt.errorbar(Moyenne,n,xerr=sem, ecolor='black')
    print(file_path)
    print(Moyenne)
    
# plt.figure()
# plt.eventplot(RASTER, lineoffsets = 3, linelengths = 3, colors = 'black')
# plt.scatter(RASTER, N, color='black')
# plt.errorbar(RASTER,N,SEM)
plt.title('Peri Stimulus Histogramme')
plt.xlabel('Temps en ms')
plt.ylabel("Nombre d'événements")
plt.axvspan(0, 5, alpha=0.1, color='blue')
plt.xlim(0,12)
plt.ylim(-0,11)
# plt.savefig("D:/Analyses/Fig 8/Figure/Premier spike/RASTERMoy2.svg", transparent=True)
