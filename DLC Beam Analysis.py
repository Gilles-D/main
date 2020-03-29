# -*- coding: utf-8 -*-
"""
@author: Gil

Read all csv of a folder (Files_path)
Calculate the time the mouse takes to cross the beam (between the start and the end point)
Plot trajectory of each trial
Calculate mean passing time for each session
Plot learning graph
Create an excel file with the time for each trial, means and std of each session
"""

import numpy as np 
from matplotlib import pyplot as plt 
import pandas as pd
import os

"""
Parameters :
    - Files_path = folder containing csv files to analyze
    - likelihood_value = DLC likelihood threshold 
    - frequency = capture frequency
    
    Bodypart :
        -19 = start x
        -22 = end x
        -13 = hindpaw x
        -16 = tail_base x
"""

Files_path = r'C:\Users\Gil\Desktop\Analyses 29-03'
likelihood_value = 0.9
frequency = 100
bdy = 16 #bodypart

writer = pd.ExcelWriter('{}/Analysis.xlsx'.format(Files_path), engine='xlsxwriter')

data_animal = []
data_session=[]
data_trial=[]
data_csv=[]
passing_times=[]
crossing_idx=[]


"""File loading list loop"""
Files = []
for r, d, f in os.walk(Files_path):
# r=root, d=directories, f = files
    for filename in f:
        if '.csv' in filename:
            Files.append(os.path.join(r, filename))


"""Work dataframe (df_data) creation"""  
for File in Files:  
    Session = 0 #Reset indexes
        
    df_raw = pd.read_csv(File) #Load file
    
    name = os.path.split(File) # Split name from path to get classification values
    name = name[1].split('.', )
    name = name[0].split()
    
    #Determine session number
    if name[5]=='J1' and name[3]=='12mm' and name[4]=='C':
        Session = 1
    if name[5]=='J1' and name[3]=='10mm' and name[4]=='C':
        Session = 2
    if name[5]=='J2' and name[3]=='10mm' and name[4]=='C':
        Session = 3
    if name[5]=='J2' and name[3]=='10mm' and name[4]=='R':
        Session = 4
    if name[5]=='J3' and name[3]=='10mm' and name[4]=='R':
        Session = 5
    if name[5]=='J4' and name[3]=='10mm' and name[4]=='R':
        Session = 6
    if name[5]=='J4' and name[3]=='8mm' and name[4]=='R':
        Session = 7
    if name[5]=='J5' and name[3]=='8mm' and name[4]=='R':
        Session = 8
    if name[5]=='J5' and name[3]=='6mm' and name[4]=='R':
        Session = 9
    
    data_animal.append(name[2])
    data_session.append(Session)
    data_trial.append(name[6])
    data_csv.append(df_raw)
df_data = pd.DataFrame({'Animal' : data_animal, 'Session' :  data_session, 'Trial' : data_trial, 'csv' : data_csv})

"""Passing times calculation - Trajectory plot"""
for i in range(len(df_data)):
    #Reset indexes
    index = 0
    row = 0
    crossing_end_x =0
    crossing_end_idx = 0
    lignes_delete=[]
    traj=0
    
    #Read identification
    a = df_data.iloc[i][0]
    s = df_data.iloc[i][1]
    t = df_data.iloc[i][2]
    data = df_data.iloc[i][3]
    
    #Determine X coords of start and end using median of both whole columns
    starting_mark_x = data.iloc[2:, 19].median()
    ending_mark_x = data.iloc[2:, 22].median()
    
    # Delete row with likelihood < threshold
    for index, row in data.iloc[2:].iterrows():
        if float(data.iloc[index, bdy+2]) < likelihood_value:
            lignes_delete.append(index)
    data = data.drop(lignes_delete)
    data = data.reset_index(drop=True)
    
    
    """
    # Trajectory plotting
    if not os.path.exists("{}\Trajectories".format(Files_path)):
        os.makedirs("{}\Trajectories".format(Files_path))
    traj = plt.figure()
    plt.scatter([float(w) for w in data.iloc[2:, bdy].tolist()],[float(w) for w in data.iloc[2:, bdy+1].tolist()])
    plt.axvline(x=starting_mark_x), plt.axvline(x=ending_mark_x)
    traj.savefig("{}\Trajectories\{}_{}_{}.pdf".format(Files_path,a,s,t))
    plt.close(traj)
    """
    
    
    # Check when bodypart crosses starting mark
    for index, row in data.iloc[2:].iterrows():
        if float(data.iloc[index,bdy]) >= starting_mark_x:
            crossing_start_x = data.iloc[index, bdy]
            crossing_start_idx = data.iloc[index, 0]
            break

    # Check when bodypart crosses ending mark
    for index, row in data.iloc[2:].iterrows():
        if float(data.iloc[index,bdy]) >= ending_mark_x:
            crossing_end_x = data.iloc[index, bdy]
            crossing_end_idx = data.iloc[index, 0]
            break
    
    #Translates idexes in time using frequency set in parameters   
    passing_times.append((float(crossing_end_idx)-float(crossing_start_idx)) /frequency)
    crossing_idx.append([crossing_start_idx,crossing_end_idx])
    
df_data['Passing_Time']=passing_times
df_data['Crossing_idx']=crossing_idx
df_data_droped = df_data.drop(['csv'], axis=1)
df_data_droped.to_excel(writer, sheet_name='Analysis')

    
"""
Passing times means and std
"""
groups = df_data.groupby(['Animal','Session'])
df_groups = groups['Passing_Time'].agg([np.mean, np.std])
df_groups = df_groups.reset_index()
df_groups.to_excel(writer, sheet_name='Means')
    
"""
Learning Plot
"""
Animal = list(dict.fromkeys(df_groups.Animal.tolist()))
x=list(range(1,10))
if not os.path.exists("{}\Learning_plots".format(Files_path)):
    os.makedirs("{}\Learning_plots".format(Files_path))
for a in Animal:
    y = []
    yerr=[]
    for i in range(len(df_groups.index)):
        if df_groups.iloc[i,0] == a:
            y.append(df_groups.iloc[i,2])
            yerr.append(df_groups.iloc[i,3])
    fig = plt.figure()
    plt.errorbar(x, y, yerr,fmt='o-'), plt.ylim(0,7), plt.title("{}".format(a))
    fig.savefig("{}\Learning_plots\{}.pdf".format(Files_path,a))

#Excel File
writer.save()

