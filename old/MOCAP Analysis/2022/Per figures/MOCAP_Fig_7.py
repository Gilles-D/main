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
import seaborn as sns
from scipy.signal import savgol_filter

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

def speed(file,marker):
    pos=coord(file,marker)
    speed=[]
    
    for i in range(len(pos[0])):
        if i !=0:
            try:
                speed.append((((pos[0][i]-pos[0][i-1]))+((pos[1][i]-pos[1][i-1]))+((pos[2][i]-pos[2][i-1])))*40)
            except:
                pass # doing nothing on exception
    
    return speed


def half(array):
    indexes=[]
    a=list(range(len(array[1])))
    u=0
    
    for i in range(len(array[1])):
        
        if u< len(array[0]):
            indexes.append(a[u])
        u=u+3
    indexes.pop(-1)
    indexes_todelete=np.delete(np.array(range(len(array[1]))),indexes)
    x=array[0]
    y=-array[1]
    z=array[2]
    
    x=list(np.delete(x,indexes_todelete))
    y=list(np.delete(y,indexes_todelete))
    z=list(np.delete(z,indexes_todelete))

    return x,y,z


root_dir = r'C:\Users\Gilles.DELBECQ\Desktop\CSV_MOCAP_05_11\Octobre 2021\6567'

obstacle_csv =r"C:/Users/Gilles.DELBECQ/Desktop/CSV_MOCAP_05_11/Octobre 2021/Param Mocap Nov 2021/obstacles.xlsx"
data_csv = r'C:/Users/Gilles.DELBECQ/Desktop/CSV_MOCAP_05_11/Octobre 2021/Param Mocap Nov 2021/Data.xlsx'

df_data = pd.read_excel(data_csv)

DIST,STIM,SPOT,OBST,SESSION=[],[],[],[],[]



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
    df_obstacle = pd.read_excel(obstacle_csv)
    irbeam=df_obstacle.loc[df_obstacle['ID'] == 'beam']

    marker_list = get_marker_list(file)
    subject=marker_list[0].split(':')[0]
    session = file.split('\\')[-1].split('_')[1]
    trial = file.split('\\')[-1].split('_')[2].split('.')[0]
    

    df_data_file = df_data.loc[((df_data['Animal'] == int(subject)) & (df_data['Session'] == int(session))& (df_data['Trial'] == int(trial)))]
    freq = df_data_file['Freq'].tolist()[0]
    power = df_data_file['Power'].tolist()[0]
    tracking_quality = df_data_file['Tracking quality'].tolist()[0]
    obstacle_idx = df_data_file['Obstacle'].tolist()[0]
    
    
    
    
    
    print(f'Animal : {subject} Session : {session} Trial : {trial} Freq : {freq} Quality : {tracking_quality}')
    


    left_foot=coord(file,"{}:Foot_L2".format(subject))
    right_foot=coord(file,"{}:Foot_R2".format(subject))
    left_tibia=coord(file,"{}:Tibia_L2".format(subject))
    left_thigh=coord(file,"{}:Thigh_L2".format(subject))
    left_pelvis=coord(file,"{}:Pelvis_L2".format(subject))
    
    right_tibia=coord(file,"{}:Tibia_R2".format(subject))
    right_thigh=coord(file,"{}:Thigh_R2".format(subject))
    right_pelvis=coord(file,"{}:Pelvis_R2".format(subject))

    """
    Fig6 : Step over left_foot
    Plot the last stance cycle around the obstacle
    
    Position obstacle
    Fenetre avant/après (X points, ou coordonnées en x)
    """
    right_foot_speed=list(speed(file,"{}:Foot_R2".format(subject)))
    
    
    window = 100
    if obstacle_idx != 0:
        obstacle = df_obstacle.loc[df_obstacle['ID'] == obstacle_idx]
    
        figure6 = plt.figure()
        ax6 = figure6.add_subplot(111)
        ax6.set_aspect('equal')
        plt.title(f'Step over {subject}_{session}_{trial}')
        foot=[]
        idx_list=[]
        for i in range(len(right_foot[0])):
            if right_foot[1][i] >= float(obstacle['Y'])-window and right_foot[1][i] <= float(obstacle['Y'])+window :
                plt.plot([-right_foot[1][i],-right_tibia[1][i]],[right_foot[2][i],right_tibia[2][i]],color='deepskyblue',alpha=0.7)
                plt.plot([-right_tibia[1][i],-right_thigh[1][i]],[right_tibia[2][i],right_thigh[2][i]],color='red',alpha=0.7)
                plt.plot([-right_thigh[1][i],-right_pelvis[1][i]],[right_thigh[2][i],right_pelvis[2][i]],color='green',alpha=0.7)
                foot.append(right_foot[2][i])
                idx_list.append(i)
        
        pre_idx=idx_list[foot.index(min(foot[0:int(len(foot)/2)]))]
        post_idx=idx_list[foot.index(min(foot[int(len(foot)/2):-1]))]
        pre=-(-right_foot[1][pre_idx]+float(obstacle['Y']))
        post=-right_foot[1][post_idx]+float(obstacle['Y'])

        plt.plot([-float(obstacle['Y']),-float(obstacle['Y'])],[0,float(obstacle['Z'])],linewidth =10,color='black',alpha=0.3)
        plt.axvline(-right_foot[1][pre_idx])
        plt.axvline(-right_foot[1][post_idx])
        
        # figure6.savefig(f'{root_dir}/Figs/{subject}_{session}_{trial}_Fig6.svg')        
        plt.close()  
        
        
        # figure_speed = plt.figure()
        # plt.plot(right_foot_speed[idx_list[0]:idx_list[-1]])
        # plt.close()
        
        if freq == 'None':
            print(freq)
            DIST.append(pre)
            STIM.append('Off')
            SPOT.append('Pre')
            DIST.append(post)
            STIM.append('Off')
            SPOT.append('Post')
            OBST.append(obstacle_idx)
            SESSION.append(f'Step over {subject}_{session}_{trial}')
        else:
            DIST.append(pre)
            STIM.append('On')
            SPOT.append('Pre')
            DIST.append(post)
            STIM.append('On')
            SPOT.append('Post')
            OBST.append(obstacle_idx)
            SESSION.append(f'Step over {subject}_{session}_{trial}')
        
        step_obstacle_df = pd.DataFrame(
            {'Distance':pd.Series(DIST),
              'Spot':pd.Series(SPOT),
              'Stim':pd.Series(STIM),
              'Obstacle':pd.Series(OBST),
              'Session':pd.Series(SESSION)
                })
        


    
    """
    Fig 7 : light effect
    lire le csv data
    sélectionner l'animal
    
    ploter chaque essais l'un sur l'autre
    pour chaque ligne - trial, regarder s'il y a stim : changer de couleur
    
    
    """
# figure7 = plt.figure()    
# ax = sns.swarmplot(data=step_obstacle_df,x="Spot",y='Distance',hue='Stim',dodge='True',palette=['gray','orange'])

# step_obstacle_df2=step_obstacle_df.drop(step_obstacle_df[step_obstacle_df.Obstacle != 's8'].index)

# figure7 = plt.figure()    
# ax = sns.boxplot(data=step_obstacle_df2,x="Spot",y='Distance',hue='Stim',dodge='True',palette=['gray','orange'])