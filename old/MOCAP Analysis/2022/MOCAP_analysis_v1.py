# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 15:54:49 2021

@author: Gilles.DELBECQ

Analysis of MOCAP CSV output

v1 14/10/21



Adapted from Ludovic SPAETH

"""


import pandas as pd
from matplotlib import pyplot as plt 
import numpy as np
import os
import math
from scipy.signal import find_peaks

def get_marker_list(file):
    '''
    Gets marker list from MOCAP file 
    '''
    import pandas as pd
    
    data = pd.read_csv(file,sep=',',header=2,delimiter=None,na_values='')

    #Get markers list
    markers = [x for x in data.columns if 'Unnamed' not in x]
    
    #Assert all markers are string formated
    for marker in markers : 
        assert type(marker) == str, 'Markers are not strings'
    
    return markers

def new_file_index(file):
    '''
    Creates new index for optimized dataframe, including "Marker1:X","Marker1:Y"...format
    '''
    
    pre_format = ['Frame','SubFrame']
    
    positions = ['X','Y','Z']
    
    markers = get_marker_list(file)
    
    marker_index = []
    for marker in markers :
        for position in positions : 
            marker_index.append('{}:{}'.format(marker,position))
        
    new_file_index = pre_format + marker_index
        
    return new_file_index

def dataframe(file,header=4):
    
    '''
    Returns on optimzed dataframe based on architecture of the raw file
    '''
    
    import pandas as pd 
    
    data = pd.read_csv(file,sep=',',header=header,delimiter=None,na_values='')
    
    opt_dataframe = pd.DataFrame(data.values,columns=new_file_index(file))
    
    return opt_dataframe


def coord(file,marker,fstart=1,fstop=-1,projection=None,step=1):
    '''
    Returns array with XYZ coordinates for a single marker
    '''
   
    data = dataframe(file)
    
    if fstop == -1:
        stop = data.shape[0]-1
    else:
        stop = fstop
        
    xs = data.iloc[fstart:stop,data.columns.get_loc('{}:X'.format(marker))].values
    ys = data.iloc[fstart:stop,data.columns.get_loc('{}:Y'.format(marker))].values
    zs = data.iloc[fstart:stop,data.columns.get_loc('{}:Z'.format(marker))].values
    

    if projection == 'X':
        proj = np.arrange(0,len(xs),step)
        xs = np.asarray([x+w for x,w in zip(xs,proj)]).ravel()
    
    if projection == 'Y':
        proj = np.arange(0,len(ys),step)
        ys = np.asarray([y+w for y,w in zip(ys,proj)]).ravel()

    return xs,ys,zs

def normalized(file, ref_marker, target_marker):
    """
    Calculate the normalized coordinates of target_marker with ref_marker as the reference
    """
    ref_x,ref_y,ref_z = coord(file,ref_marker)
    target_x, target_y,target_z = coord(file,target_marker)
    norm_x=target_x-ref_x
    norm_y=target_y-ref_y
    norm_z=target_z-ref_z
    
    return norm_x,norm_y,norm_z

def calculate_angle(file,markerA,markerB,markerC):
    """
    Calculate angle formed in markerB
    https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates
    """
    A,B,C=np.asarray(coord(file,markerA)),np.asarray(coord(file,markerB)),np.asarray(coord(file,markerC))
    ba = A - B
    bc = C - B
    
    cosine_angle=[]
    for i in range(len(ba[0])):
        cosine_angle.append(np.dot(ba[:,i], bc[:,i]) / (np.linalg.norm(ba[:,i]) * np.linalg.norm(bc[:,i])))
    
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def detect_stance(file,marker):
    trajectory=coord(file,marker)[2]
    acceleration =np.gradient(np.gradient(trajectory))
    peaks, _ = find_peaks(acceleration,prominence=1)
    
    return trajectory,acceleration,peaks


obstacles = [(0.32,72,33.6),(1.31,104.8,55.4),(2.05,160,45.1)]
obstacles = [(-2.75,130,11.62),(1.31,104.8,55.4),(2.05,160,45.1)]
obstacles=(-2.75,130,11.62)



root_dir = r'C:\Users\Gilles.DELBECQ\Desktop\CSV_MOCAP\Octobre 2021\6464'

#First Loop : loop on all csv files to list them in the list "Files"
Files = []
for r, d, f in os.walk(root_dir):
# r=root, d=directories, f = files
    for filename in f:
        if '.csv' in filename:
            Files.append(os.path.join(r, filename))
            
print('Files to analyze : {}'.format(len(Files)))

for file in Files:
    df_raw = pd.read_csv(file,sep=',',header=2,delimiter=None,na_values='')

    marker_list = get_marker_list(file)
    subject=marker_list[0].split(':')[0]
    session = file.split('\\')[-1].split('_')[1]
    trial = file.split('\\')[-1].split('_')[2].split('.')[0]

    """
    Fig1 : Normalized feet trajectory from reference
    
    """
    norm_left_foot = normalized(file, "{}:Thigh_L2".format(subject), "{}:Foot_L2".format(subject))
    norm_right_foot = normalized(file, "{}:Thigh_R2".format(subject), "{}:Foot_R2".format(subject))
    
    figure = plt.figure()
    plt.title('Normalized feet trajectory')
    plt.legend()
    plt.plot(-norm_left_foot[1],norm_left_foot[2],color='red',label='Left')
    plt.plot(-norm_right_foot[1],norm_right_foot[2],color='blue',label='Right')




    """
    Fig2 : Raw Feet trajectory
    """
    left_foot=coord(file,"{}:Foot_L2".format(subject))
    right_foot=coord(file,"{}:Foot_R2".format(subject))
    
    figure2 = plt.figure()
    plt.title('Raw Feet trajectory')
    plt.legend()
    plt.plot(-left_foot[1],left_foot[2],color='red',label='Left')
    plt.plot(-right_foot[1],right_foot[2],color='blue',label='Right')
    
    # plt.plot([obstacles[0][1],obstacles[0][1]],[obstacles[0][2],5.5],linewidth =10,color='black',alpha=0.3)
    # plt.plot([obstacles[1][1],obstacles[1][1]],[obstacles[1][2],5.5],linewidth =10,color='black',alpha=0.3)
    # plt.plot([obstacles[2][1],obstacles[2][1]],[obstacles[2][2],5.5],linewidth =10,color='black',alpha=0.3)




    """
    Fig3 : full leg trajectory
    """
    left_tibia=coord(file,"{}:Tibia_L2".format(subject))
    left_thigh=coord(file,"{}:Thigh_L2".format(subject))
    left_pelvis=coord(file,"{}:Pelvis_L2".format(subject))
    
    figure3 = plt.figure()
    plt.title('Full leg trajectory')
    plt.plot([list(-left_foot[1]),list(-left_tibia[1])],[list(left_foot[2]),list(left_tibia[2])],color='deepskyblue')
    plt.plot([list(-left_tibia[1]),list(-left_thigh[1])],[list(left_tibia[2]),list(left_thigh[2])],color='red')
    plt.plot([list(-left_thigh[1]),list(-left_pelvis[1])],[list(left_thigh[2]),list(left_pelvis[2])],color='green')
    
    # plt.plot([obstacles[0][1],obstacles[0][1]],[obstacles[0][2],5.5],linewidth =10,color='black',alpha=0.3)
    # plt.plot([obstacles[1][1],obstacles[1][1]],[obstacles[1][2],5.5],linewidth =10,color='black',alpha=0.3)
    # plt.plot([obstacles[2][1],obstacles[2][1]],[obstacles[2][2],5.5],linewidth =10,color='black',alpha=0.3)



    """
    Fig4 : angles left_foot
    """
    fig, axs = plt.subplots(3,sharex=True)
    fig.suptitle('Angles')
    axs[0].plot(calculate_angle(file,"{}:Thigh_L2".format(subject),"{}:Pelvis_L2".format(subject),"{}:Back1".format(subject)))
    axs[0].set_title('Hip')
    axs[1].plot(calculate_angle(file,"{}:Tibia_L2".format(subject),"{}:Thigh_L2".format(subject),"{}:Pelvis_L2".format(subject)))
    axs[1].set_title('Knee')
    axs[2].plot(calculate_angle(file,"{}:Foot_L2".format(subject),"{}:Tibia_L2".format(subject),"{}:Thigh_L2".format(subject)))
    axs[2].set_title('Ankle')


    """
    Fig5 : Stances
    
    """
    
    # trajectory,acceleration,peaks = detect_stance(file, "{}:Foot_L2".format(subject))
    
    # figure5 = plt.figure()
    # plt.title('Stances cycle')
    # plt.plot(trajectory)
    # plt.plot(acceleration)
    # plt.plot(peaks, acceleration[peaks], "x")

    
    """
    Fig6 : Step over left_foot
    Plot the last stance cycle around the obstacle
    
    Position obstacle
    Fenetre avant/après (X points, ou coordonnées en x)
    """
    
    figure6 = plt.figure()
    plt.title('Step over')

    #obstacle
    
    window = 150
    obstacle=(-2.75,130,11.62)
    
    for i in range(len(left_foot[0])):
        if left_foot[1][i] >= obstacle[1]-window and left_foot[1][i] <= obstacle[1]+window :
            plt.plot([-left_foot[1][i],-left_tibia[1][i]],[left_foot[2][i],left_tibia[2][i]],color='deepskyblue')
            plt.plot([-left_tibia[1][i],-left_thigh[1][i]],[left_tibia[2][i],left_thigh[2][i]],color='red')
            plt.plot([-left_thigh[1][i],-left_pelvis[1][i]],[left_thigh[2][i],left_pelvis[2][i]],color='green')


