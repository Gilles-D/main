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
Datapath1 = 'C:/Users\gille\Desktop\Analyses\Baseline/data Séparé/Rep'
Datapath2 = 'C:/Users\gille\Desktop\Analyses\Baseline/data Séparé/Rep Pas'

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
    X = pd.read_excel(file_path, usecols='C')*1000
    Y = pd.read_excel(file_path, usecols='B')*1000
    # X = np.average(pd.read_excel(file_path, usecols='C'))*1000
    # Y = np.average(pd.read_excel(file_path, usecols='B'))*1000
    """
    # xerr = stats.sem(pd.read_excel(file_path, usecols='C')*1000, ddof=len(pd.read_excel(file_path, usecols='B'))-2)
    # yerr = stats.sem(pd.read_excel(file_path, usecols='B')*1000, ddof=len(pd.read_excel(file_path, usecols='B'))-2)
    # # pi = math.pi
    # # area = xerr*yerr*pi
    # # plt.scatter(x, y, s=area)
    # plt.errorbar(x,y,xerr,yerr)
    # X = [x, x-xerr, x+xerr, x, x]
    # Y = [y, y, y, y-yerr, y+yerr]
    """
    plt.scatter(X, Y, color='blue')
    print (manip)

for file_path in List_File_paths2:
    manip = os.path.splitext(os.path.basename(file_path))[0]
    reponse = os.path.basename(os.path.dirname(file_path))
    X = pd.read_excel(file_path, usecols='C')*1000
    Y = pd.read_excel(file_path, usecols='B')*1000
    # X = np.average(pd.read_excel(file_path, usecols='C'))*1000
    # Y = np.average(pd.read_excel(file_path, usecols='B'))*1000
    plt.scatter(X, Y, color='orange')
    print (manip)

plt.title('Scatter plot')
plt.xlabel('Durée demi pic')
plt.ylabel('Intervalle Pic Max-Pic Min')
plt.xlim(0, 1.2)
plt.ylim(0, 1.7)
plt.show()
