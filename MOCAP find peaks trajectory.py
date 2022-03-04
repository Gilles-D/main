# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 15:14:13 2022

@author: Gilles.DELBECQ
"""



import pandas as pd
from matplotlib import pyplot as plt 
import numpy as np
import os
import math
from scipy.signal import find_peaks , peak_prominences
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

ANIMAL,SESSION,TRIAL,PEAKS_X,PEAKS_Y=[],[],[],[],[]

Hauteurs=[]

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
    
    right_foot=coord(file,"{}:Foot_R2".format(subject))
    z = right_foot[2]
    peaks, _ = find_peaks(z,prominence=5)
    plt.figure()
   
    plt.plot(-right_foot[1],right_foot[2],color='blue',label='Right')
    plt.plot(-right_foot[1][peaks], right_foot[2][peaks], "x")
    plt.title(f'Animal : {subject} Session : {session} Trial : {trial}')
    
    if session == '1' or session =='5':
        for i in peaks:
            ANIMAL.append(subject)
            SESSION.append(session)
            TRIAL.append(trial)
            PEAKS_X.append(i)
            PEAKS_Y.append(-right_foot[1][i])
    
    # ANIMAL.append(subject)
    # SESSION.append(session)
    # TRIAL.append(trial)
    # PEAKS.append(list(zip(list(peaks),list(-right_foot[1][peaks]))))
    
save_df = pd.DataFrame(
    {'ANIMAL':pd.Series(ANIMAL),
      'SESSION':pd.Series(SESSION),
      'TRIAL':pd.Series(TRIAL),
      'PEAKS_X':pd.Series(PEAKS_X),
      'PEAKS_Y':pd.Series(PEAKS_Y)
        })

ax = sns.swarmplot(x="PEAKS_X", y="PEAKS_Y", data=save_df)

# for index, row in save_df.iterrows():
#     for i in row['PEAKS']:
#         print(i[1])