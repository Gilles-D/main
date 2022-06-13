# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 15:54:49 2021

@author: Gilles.DELBECQ

Analysis of MOCAP CSV output


"""


import pandas as pd
from matplotlib import pyplot as plt 
import numpy as np


def get_bodypart_index(df_raw, bodypart):
    return df_raw.columns[(df_raw.values==bodypart).any(0)].tolist()[0]
    

def substract_ref(df_raw,Bdy_ref,Bdy_track):
    x,y,z = [],[],[]
    for i in range(len(df_raw.iloc[3:,get_bodypart_index(df_raw,Bdy_ref)])):
        x.append(float((df_raw.iloc[3:,get_bodypart_index(df_raw,Bdy_track)]).iloc[i])-float((df_raw.iloc[3:,get_bodypart_index(df_raw,Bdy_ref)]).iloc[i]))
        y.append(float((df_raw.iloc[3:,get_bodypart_index(df_raw,Bdy_track)+1]).iloc[i])-float((df_raw.iloc[3:,get_bodypart_index(df_raw,Bdy_ref)+1]).iloc[i]))
        z.append(float((df_raw.iloc[3:,get_bodypart_index(df_raw,Bdy_track)+2]).iloc[i])-float((df_raw.iloc[3:,get_bodypart_index(df_raw,Bdy_ref)+2]).iloc[i]))
        
        # df_susbtracted = pd.DataFrame(list(zip(x,y,z)), columns =['x', 'y',"z"])
        df_susbtracted = np.array((x, y, z), dtype=float)
        
    return df_susbtracted

def leg_trace(df_raw,leg)


File = "C:/Users/Gilles.DELBECQ/Desktop/CSV_MOCAP/Thy1Cop4 Cal 05.csv"
df_raw = pd.read_csv(File, sep=',', header=None,skiprows=2)

Bodyparts = list(df_raw.iloc[0].dropna())
left_leg=()
right_leg=()


left = substract_ref(df_raw,'Mouse 6 MArkers:Back_Tail','Mouse 6 MArkers:Ankle_Left')
right = substract_ref(df_raw,'Mouse 6 MArkers:Back_Tail','Mouse 6 MArkers:Ankle_Right')

fig = plt.figure()
plt.plot(left[1],left[2],label='Left', color='blue')
plt.plot(right[1],right[2],label='Right', color='red')
fig.legend()
