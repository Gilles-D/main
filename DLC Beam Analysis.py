# -*- coding: utf-8 -*-
"""
@author: Gil

Video file naming : XXX XXX ANIMAL SIZE SHAPE(C/R) DAY TRIAL

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
import math

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
        -4 = eye x
"""

Files_path = r'C:\Users\Gil\Desktop\Analyses 02-04'
likelihood_value = 0.9
frequency = 100
bdy_pt = 16 #bodypart passing time
bdy_tr = 13 #bodypart trajectory
bdy_is = 13

writer = pd.ExcelWriter('{}/Analysis.xlsx'.format(Files_path), engine='xlsxwriter')

data_animal = []
data_session=[]
data_trial=[]
data_csv=[]
passing_times=[]
crossing_idx=[]

def curve(x, y):
    return y-(-0.00004892765485336647*(x**2))-(0.082683853358223*x)+511.84437126819443
def y_curve(x):
    return (-0.00004892765485336647*(x**2))-(0.082683853358223*x)+511.84437126819443

def calculateDistance(x1,y1,x2,y2):  
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist

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

"""Passing times calculation - Trajectory plot - Instantaneous speed"""
for i in range(len(df_data)):
    #Reset indexes
    index = 0
    row = 0
    crossing_end_x =0
    crossing_end_idx = 0
    traj=0
    
    #Read identification
    a = df_data.iloc[i][0]
    s = df_data.iloc[i][1]
    t = df_data.iloc[i][2]
    data = df_data.iloc[i][3]
    
    #Determine X coords of start and end using median of both whole columns
    starting_mark_x = data.iloc[2:, 19].median()
    ending_mark_x = data.iloc[2:, 22].median()
    starting_mark_y = data.iloc[2:, 20].median()
    ending_mark_y = data.iloc[2:, 23].median()
    
    """ Instantaneous speed """       
    IS_index = []
    IS_speed=[]
    IS_idx = []
        
    for w in data.index[2:]:
        if float(data.iloc[w, bdy_is+2])>=likelihood_value:
            IS_index.append(w)
    
    for w in range(len(IS_index)):
        if float(w) == 0:
            pass
        else:  
            distance = (((calculateDistance(float(data.iloc[IS_index[w-1], bdy_is]), float(data.iloc[IS_index[w-1], bdy_is+1]), float(data.iloc[IS_index[w], bdy_is]), float(data.iloc[IS_index[w], bdy_is+1])))*52)/calculateDistance(starting_mark_x, starting_mark_y, ending_mark_x, ending_mark_y))
            time = (IS_index[w]-(IS_index[w-1]))*(1/frequency)
            IS_idx.append(data.iloc[w, bdy_tr])
            IS_speed.append(distance/time)    
    
    """Subplot avec trajectoire"""
    if not os.path.exists("{}\Trajectories".format(Files_path)):
        os.makedirs("{}\Trajectories".format(Files_path))
    if not os.path.exists("{}\Trajectories\{}".format(Files_path, a)):
        os.makedirs("{}\Trajectories\{}".format(Files_path, a))
    traj, subplot = plt.subplots(2,1)
    subplot[0].set_title("Instantaneous speed {} {} {}".format(a, s, t))
    subplot[0].plot(IS_speed, color='crimson') 
    subplot[0].xaxis.set_visible(False)
    subplot[0].set_ylabel("Speed (cm/s)")
        
    """ Trajectory plotting """
    subplot[1].scatter([starting_mark_x, ending_mark_x], [curve(starting_mark_x, starting_mark_y),curve(ending_mark_x, ending_mark_y)], color='crimson', marker='|')
    subplot[1].plot([starting_mark_x, ending_mark_x], [curve(starting_mark_x, starting_mark_y)+6,curve(ending_mark_x, ending_mark_y)+6], color='black', linewidth=1)
    subplot[1].plot([starting_mark_x, ending_mark_x], [curve(starting_mark_x, starting_mark_y)-6,curve(ending_mark_x, ending_mark_y)-6], color='black', linewidth=1)
    subplot[1].plot([float(data.iloc[w, bdy_tr]) for w in data.index[2:] if float(data.iloc[w, bdy_tr+2])>=likelihood_value],
         [curve(float(data.iloc[w, bdy_tr]),float(data.iloc[w, bdy_tr+1]))        
         for w in data.index[2:] if float(data.iloc[w, bdy_tr+2])>=likelihood_value], color='crimson') 
    subplot[1].set_xlabel("X Coord"), subplot[1].set_ylabel("Y Coord"), subplot[1].set_title("Trajectory")
    subplot[1].set_ylim(curve(starting_mark_x,starting_mark_y)-50,curve(starting_mark_x,starting_mark_y)+50)
    subplot[1].invert_yaxis()
    traj.savefig("{}\Trajectories\{}\{}_{}_{}.svg".format(Files_path,a,a,s,t))
    plt.close(traj)

    """ Passing Time """
    # Check when bodypart crosses starting mark
    for index, row in data.iloc[2:].iterrows():
        if float(data.iloc[index,bdy_pt]) >= starting_mark_x and float(data.iloc[index, bdy_pt+2])>=likelihood_value:
            crossing_start_x = data.iloc[index, bdy_pt]
            crossing_start_idx = data.iloc[index, 0]
            break

    # Check when bodypart crosses ending mark
    for index, row in data.iloc[2:].iterrows():
        if float(data.iloc[index,bdy_pt]) >= ending_mark_x and float(data.iloc[index, bdy_pt+2])>=likelihood_value:
            crossing_end_x = data.iloc[index, bdy_pt]
            crossing_end_idx = data.iloc[index, 0]
            break
    
    #Translates idexes in time using frequency set in parameters   
    passing_times.append((float(crossing_end_idx)-float(crossing_start_idx)) /frequency)
    crossing_idx.append([crossing_start_idx,crossing_end_idx])
df_data['Passing_Time']=passing_times
df_data['Crossing_idx']=crossing_idx
df_data_droped = df_data.drop(['csv'], axis=1)
df_data_droped.to_excel(writer, sheet_name='Analysis')

    
""" Passing times means and std """
groups = df_data.groupby(['Animal','Session'])
df_groups = groups['Passing_Time'].agg([np.mean, np.std])
df_groups = df_groups.reset_index()
df_groups.to_excel(writer, sheet_name='Means')
    
""" Learning Plot """
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
    plt.errorbar(x, y, yerr,fmt='o-', color='crimson'), plt.ylim(0,7), plt.title("Learning Plot {}".format(a)), plt.xlabel("Session"), plt.ylabel("Mean Passing Time")
    fig.savefig("{}\Learning_plots\{}.svg".format(Files_path,a))

#Excel File
writer.save()