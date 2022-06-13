# -*- coding: utf-8 -*-
"""
Read all csv of a folder (Files_path)
Calculate the time the mouse takes to cross the beam (between the start and the end point)
Create an excel file with the time for each file
"""

import numpy as np 
from matplotlib import pyplot as plt 
import scipy.signal as sp
from scipy import stats 
import pandas as pd
 
import os

"""
Parameters :
    - Files_path = folder containing csv files to analyze
    - confidence_value = DLC likelihood threshold 
    - frequency = capture frequency
    
    
    Bodypart :
        -19 = start x
        -22 = end x
        -13 = hindpaw x
        - 15 = hindpaw likelihood
        -16 = tail_base x
        -18 = tail_base likelihood
"""

Files_path = 'C:/Users/Gilles.DELBECQ/Desktop/J1'
likelihood_value = 0.7
frequency = 100

#File loading list loop
Files = []
data_filename = []
data_passing_time = []
for r, d, f in os.walk(Files_path):
# r=root, d=directories, f = files
    for filename in f:
        if '.csv' in filename:
            Files.append(os.path.join(r, filename))
            
for File in Files:
    index = 0
    row = 0
    crossing_end_x =0
    crossing_end_idx = 0
    df = pd.read_csv(File)
    starting_mark_x = df.iloc[4:, 19].median()
    ending_mark_x = df.iloc[4:, 22].median()
    
    for index, row in df.iloc[4:].iterrows():
        if float(df.iloc[index-1:index+1, 13].median()) >= starting_mark_x and float(df.iloc[index-1:index+1, 13].median()) <= ending_mark_x and float(df.iloc[index-1, 15]) >= likelihood_value and float(df.iloc[index, 15]) >= likelihood_value and float(df.iloc[index+1, 15]) >= likelihood_value:
            crossing_start_x = df.iloc[index, 13]
            crossing_start_idx = index
            break
    index = 0
    row = 0
    for index, row in df.iloc[crossing_start_idx:].iterrows():
        if float(df.iloc[index-1:index+1, 13].median()) >= ending_mark_x and float(df.iloc[index-1, 15]) >= likelihood_value and float(df.iloc[index, 15]) >= likelihood_value and float(df.iloc[index+1, 15]) >= likelihood_value:
            crossing_end_x = df.iloc[index, 13]
            crossing_end_idx = index
            break
        
    passing_time = (crossing_end_idx-crossing_start_idx) /frequency
    
    name = os.path.split(File)
    name = name[1].split('.', )
    print(name[0])
    print(starting_mark_x, ending_mark_x)
    print(crossing_start_x, crossing_end_x)
    print(crossing_start_idx/frequency, crossing_end_idx/frequency)
    print(crossing_end_idx)
    print(crossing_end_x)
    print(passing_time)

    data_filename.append(name[0])
    data_passing_time.append(passing_time)

data = {'File Name' : data_filename, 'Passing Time' : data_passing_time}
data_df = pd.DataFrame(data)
data_df.to_excel('{}/passing_time.xlsx'.format(Files_path))