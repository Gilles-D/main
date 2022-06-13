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

class MOCAP_file:
    
    def __init__(self,filepath):
        import pandas as pd
        self.df_raw = pd.read_csv(filepath,sep=',',header=2,delimiter=None,na_values='')
        self.filepath=filepath
    
    
    def get_marker_list(self):
        '''
        Gets marker list from MOCAP file 
        '''

        #Get markers list
        markers = [x.split(':')[-1] for x in self.df_raw.columns if 'Unnamed' not in x]
                
        #Assert all markers are string formated
        for marker in markers : 
            assert type(marker) == str, 'Markers are not strings'
        
        return markers

    def get_marker_list(self):
        '''
        Gets marker list from MOCAP file 
        '''

        #Get markers list
        markers = [x.split(':')[-1] for x in self.df_raw.columns if 'Unnamed' not in x]
                
        #Assert all markers are string formated
        for marker in markers : 
            assert type(marker) == str, 'Markers are not strings'
        
        return markers

    def subject(self):
        return str(self.filepath.split('/')[-1].split('_')[0])
    
    def session_idx(self):
        return self.filepath.split('/')[-1].split('_')[1]
        
    def trial_idx(self):
        return self.filepath.split('/')[-1].split('_')[2].split('.')[0]     


    def new_file_index(self):
        '''
        Creates new index for optimized dataframe, including "Marker1:X","Marker1:Y"...format
        '''
        
        pre_format = ['Frame','SubFrame']
        
        positions = ['X','Y','Z']
        
        markers = [x for x in self.df_raw.columns if 'Unnamed' not in x]

        marker_index = []
        for marker in markers :
            for position in positions : 
                marker_index.append('{}:{}'.format(marker,position))
 
        new_file_index = pre_format + marker_index
                   
        return new_file_index
    
    def dataframe(self,header=4):
        
        '''
        Returns on optimzed dataframe based on architecture of the raw file
        '''
        data = pd.read_csv(self.filepath,sep=',',header=header,delimiter=None,na_values='')
        
        opt_dataframe = pd.DataFrame(data.values,columns=self.new_file_index())
        return opt_dataframe
    
    
    def coord(self,marker,fstart=1,fstop=-1,projection=None,step=1):
        '''
        Returns array with XYZ coordinates for a single marker
        '''
              
        data=self.dataframe()
        
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
    
    def normalized(self, ref_marker, target_marker):
        """
        Calculate the normalized coordinates of target_marker with ref_marker as the reference
        """
        ref_x,ref_y,ref_z = self.coord(ref_marker)
        target_x, target_y,target_z = self.coord(target_marker)
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



file_path = 'D:/Working_Dir/MOCAP/Fev2022/Raw_CSV/1111/1111_01_17.csv'
data_MOCAP = MOCAP_file(file_path)



# left_foot=data_MOCAP.coord(f"{data_MOCAP.subject()}:Left_Foot")

left_foot=data_MOCAP.coord("New Subject:Left_Foot")
right_foot=data_MOCAP.coord("New Subject:Right_Foot")

figure1 = plt.figure()

norm_left_foot = data_MOCAP.normalized("New Subject:Back1", "New Subject:Left_Foot")
norm_right_foot = data_MOCAP.normalized("New Subject:Back1", "New Subject:Right_Foot")

# plt.title(f'Normalized feet trajectory {subject}_{session}_{trial}')
# ax1 = figure.add_subplot(111)

plt.plot(-norm_left_foot[1],norm_left_foot[2],color='red',label='Left')
plt.plot(-norm_right_foot[1],norm_right_foot[2],color='blue',label='Right')

a,b,c,d=[],[],[],[]
right_foot=coord(file,"{}:Foot_R2".format(subject))
Back2 = coord(file,"{}:Back2".format(subject))

# for i in range(len(norm_right_foot[1])):
#     if -Back2[1][i] <= 20.6:
#         a.append(-norm_right_foot[1][i])
#         b.append(norm_right_foot[2][i])

#     else:
#         c.append(-norm_right_foot[1][i])
#         d.append(norm_right_foot[2][i])

# plt.plot(a,b,color='blue',label='Right')
# plt.plot(c,d,color='cyan',label='RightS')

# ax1.set_aspect('equal')

plt.xlim([-25,30])
plt.ylim([-35,0])
# plt.close()

figure2 = plt.figure()
plt.title(f'Raw Feet trajectory {data_MOCAP.subject()}_{data_MOCAP.session_idx()}_{data_MOCAP.trial_idx()}')
plt.plot(-left_foot[1],left_foot[2],color='red',label='Left')
plt.plot(-right_foot[1],right_foot[2],color='blue',label='Right')


"""
TO DO
Fixer le probleme avec le nom du subject (premiers fichiers de la session 1 = New Subject)
"""
