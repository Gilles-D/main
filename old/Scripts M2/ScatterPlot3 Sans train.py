# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:27:31 2019

@author: gille


Lit l'ensemble des fichiers des dossiers repond et répond pas
Représente sur un scatter plot les critères de l'ensemble des spike (ou l'average) : rép en bleu et rep pas en orange
"""

from matplotlib import pyplot as plt 
import pandas as pd 
import numpy as np
import os
from scipy import stats 
import math
import glob

#Boucle importer fichier

#Files location
Datapath1 = 'D:/Analyses\Baseline\Data SpikeSansTrain Séparé\Rep'
Datapath2 = 'D:/Analyses\Baseline\Data SpikeSansTrain Séparé\Rep pas'

List_File_paths1 = []
for r, d, f in os.walk(Datapath1):
# r=root, d=directories, f = files
    for filename in f:
        if '.xls' in filename:
            List_File_paths1.append(os.path.join(r, filename))

List_File_paths2 = []
for r, d, f in os.walk(Datapath2):
# r=root, d=directories, f = files
    for filename in f:
        if '.xls' in filename:
            List_File_paths2.append(os.path.join(r, filename))

for file_path in List_File_paths1:
    manip = os.path.splitext(os.path.basename(file_path))[0]
    reponse = os.path.basename(os.path.dirname(file_path))
    X = pd.read_excel(file_path, usecols='D')
    Y = pd.read_excel(file_path, usecols='C')
    # X = np.mean(pd.read_excel(file_path, usecols='D'))
    # Y = np.mean(pd.read_excel(file_path, usecols='C'))
    plt.scatter(X, Y,s=4, color='blue')

    # xerr = stats.sem(pd.read_excel(file_path, usecols='D'), ddof=len(pd.read_excel(file_path, usecols='B'))-2)
    # yerr = stats.sem(pd.read_excel(file_path, usecols='C'), ddof=len(pd.read_excel(file_path, usecols='B'))-2)
    # plt.errorbar(X,Y,xerr,yerr, ecolor='blue')

    print (manip)

for file_path in List_File_paths2:
    manip = os.path.splitext(os.path.basename(file_path))[0]
    reponse = os.path.basename(os.path.dirname(file_path))
    X = pd.read_excel(file_path, usecols='D')
    Y = pd.read_excel(file_path, usecols='C')
    # X = np.mean(pd.read_excel(file_path, usecols='D'))
    # Y = np.mean(pd.read_excel(file_path, usecols='C'))
    plt.scatter(X, Y, s=4, color='orange')
    # xerr = stats.sem(pd.read_excel(file_path, usecols='D'), ddof=len(pd.read_excel(file_path, usecols='B'))-2)
    # yerr = stats.sem(pd.read_excel(file_path, usecols='C'), ddof=len(pd.read_excel(file_path, usecols='B'))-2)
    # plt.errorbar(X,Y,xerr,yerr, ecolor='orange')
    print (manip)

plt.title('Scatter plot')
plt.xlabel('Durée demi pic')
plt.ylabel('Intervalle Pic Max-Pic Min')
plt.xlim(0, 1)
plt.ylim(0, 1.4)
plt.show()
