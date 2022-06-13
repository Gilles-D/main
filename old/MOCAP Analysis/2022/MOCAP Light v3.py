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

from scipy.interpolate import interp1d

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


def interpolate(file,marker):
    x,y=[],[]
    bdy_part=coord(file,marker)
    
    f = interp1d(list(-bdy_part[1]),list(bdy_part[2]),kind='linear')

    l = [i for i in np.arange(-100,300,1)]
    
    for i in l:
        try:
            y.append(f(i))
            x.append(i)
        except:
            y.append(np.nan)
            x.append(i)
    
    # for i in range(-100,300):
    #     if i >= -bdy_part[1][0] and i <= -bdy_part[1][-1] and i!=-bdy_part[1][-1]:
    #         try:
    #             print(i,f(i))
    #             x.append(i)
    #             y.append(f(i))
    #         except:
    #             pass
    
    return x,y


# def interpolate(file,marker):
#     x,y=[],[]
#     bdy_part=coord(file,marker)
    
#     f = interp1d(list(-bdy_part[1]),list(bdy_part[2]),kind='linear')

    
#     for i in range(len(bdy_part[1])-1):
#         if i !=0 and i!=-bdy_part[1][-1] and math.isnan(-bdy_part[1][i]) == False:
#             try:
#                 print(i,int(round(-bdy_part[1][i])),f(int(round(-bdy_part[1][i]))))
#                 x.append(int(round(-bdy_part[1][i])))
#                 y.append(f(int(round(-bdy_part[1][i]))))
#             except:
#                 pass
    
#     return x,y


root_dir = r'C:\Users\Gilles.DELBECQ\Desktop\CSV_MOCAP_05_11\Octobre 2021\6567'

obstacle_csv =r"C:/Users/Gilles.DELBECQ/Desktop/CSV_MOCAP_05_11/Octobre 2021/Param Mocap Nov 2021/obstacles.xlsx"
data_csv = r'C:/Users/Gilles.DELBECQ/Desktop/CSV_MOCAP_05_11/Octobre 2021/Param Mocap Nov 2021/Data.xlsx'

df_data = pd.read_excel(data_csv)

X=list(range(-100,300))

df_interpolated = pd.DataFrame(X)
df_interpolated_stim = pd.DataFrame(X)


#First Loop : loop on all csv files to list them in the list "Files"
Files = []
for r, d, f in os.walk(root_dir):
# r=root, d=directories, f = files
    for filename in f:
        if '.csv' in filename:
            Files.append(os.path.join(r, filename))
            
print('Files to analyze : {}'.format(len(Files)))
df_obstacle = pd.read_excel(obstacle_csv)


for idx in df_obstacle['ID'].tolist():
    print(idx)
    figure2 = plt.figure()
    plt.title(f'Interpolated {idx}')
    df_interpolated = pd.DataFrame(X)
    df_interpolated_stim = pd.DataFrame(X)  
    
    for file in Files:
        df_raw = pd.read_csv(file,sep=',',header=2,delimiter=None,na_values='')  
        marker_list = get_marker_list(file)
        subject=marker_list[0].split(':')[0]
        session = file.split('\\')[-1].split('_')[1]
        trial = file.split('\\')[-1].split('_')[2].split('.')[0]
        
    
        df_data_file = df_data.loc[((df_data['Animal'] == int(subject)) & (df_data['Session'] == int(session))& (df_data['Trial'] == int(trial)))]
        freq = df_data_file['Freq'].tolist()[0]
        power = df_data_file['Power'].tolist()[0]
        tracking_quality = df_data_file['Tracking quality'].tolist()[0]
        obstacle_idx = df_data_file['Obstacle'].tolist()[0]

        right_foot=coord(file,"{}:Foot_R2".format(subject))

        
        if obstacle_idx == idx:
            print(f'Animal : {subject} Session : {session} Trial : {trial} Freq : {freq} obstacle : {obstacle_idx}')
            x,y=interpolate(file,"{}:Foot_R2".format(subject))
            if freq == 'None':
                plt.plot(x,y,color='black',label='Right',alpha=0.1)
                df_interpolated[f'{subject}_{session}_{trial}']=y

            else:
                plt.plot(x,y,color='orange',label='Right',alpha=0.1)
                df_interpolated_stim[f'{subject}_{session}_{trial}']=y
    obstacle = df_obstacle.loc[df_obstacle['ID'] == idx]
    # plt.plot([-float(obstacle['Y']),-float(obstacle['Y'])],[0,float(obstacle['Z'])],linewidth =10,color='black',alpha=0.3)    


    df_interpolated=df_interpolated.set_index(0)
    df_interpolated_stim=df_interpolated_stim.set_index(0)
    plt.plot(df_interpolated.median(axis=1),color='black')
    plt.plot(df_interpolated_stim.median(axis=1),color='orange')
    
    # figure3 = plt.figure()
    # plt.title(f'{idx}')
        
    # for file in Files:
    #     df_raw = pd.read_csv(file,sep=',',header=2,delimiter=None,na_values='')
            
    #     marker_list = get_marker_list(file)
    #     subject=marker_list[0].split(':')[0]
    #     session = file.split('\\')[-1].split('_')[1]
    #     trial = file.split('\\')[-1].split('_')[2].split('.')[0]
        
    
    #     df_data_file = df_data.loc[((df_data['Animal'] == int(subject)) & (df_data['Session'] == int(session))& (df_data['Trial'] == int(trial)))]
    #     freq = df_data_file['Freq'].tolist()[0]
    #     power = df_data_file['Power'].tolist()[0]
    #     tracking_quality = df_data_file['Tracking quality'].tolist()[0]
    #     obstacle_idx = df_data_file['Obstacle'].tolist()[0]

    #     right_foot=coord(file,"{}:Foot_R2".format(subject))

        
    #     if obstacle_idx == idx:
    #         print(f'Animal : {subject} Session : {session} Trial : {trial} Freq : {freq} obstacle : {obstacle_idx}')
            
    #         if freq == 'None':
    #             plt.plot(-right_foot[1],right_foot[2],color='red',label='Right',alpha=0.3)

    #         else:
    #             plt.plot(-right_foot[1],right_foot[2],color='blue',label='Right',alpha=0.5)

    # obstacle = df_obstacle.loc[df_obstacle['ID'] == idx]
    # plt.plot([-float(obstacle['Y']),-float(obstacle['Y'])],[0,float(obstacle['Z'])],linewidth =10,color='black',alpha=0.3)        
    # figure2.savefig(f'{root_dir}/Figs/{subject}_{idx}_effets stim full_med.svg')
        