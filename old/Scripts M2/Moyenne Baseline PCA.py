# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:01:45 2019

@author: gille
"""

from matplotlib import pyplot as plt 
import pandas as pd 
import numpy as np
import os
from scipy import stats 
import math
import glob
from os.path import basename

Datapath1 = 'D:/Analyses/Baseline/data2/'
A = []
B = []
Neurone = []
df = pd.DataFrame()


List_File_paths1 = []
for r, d, f in os.walk(Datapath1):
# r=root, d=directories, f = files
    for filename in f:
        if '.xls' in filename:
            List_File_paths1.append(os.path.join(r, filename))

           
for file_path in List_File_paths1:    
    # data = pd.read_excel(file_path, 'Sheet1')*1000
    # moy = np.mean(data)
    # df = df.append(moy,ignore_index=True)
    a = np.array(pd.read_excel(file_path, usecols='B')*1000)
    b = np.array(pd.read_excel(file_path, usecols='C')*1000)
    A.append(np.mean(a))
    B.append(np.mean(b))
    neurone = os.path.splitext(os.path.basename(file_path))[0]
    Neurone.append(neurone)

data_critere = { 'Neurone' : Neurone, 'A' : A, 'B' : B}
df = pd.DataFrame(data_critere)
df.to_excel('D:/Analyses/Baseline/moy2.xlsx')
