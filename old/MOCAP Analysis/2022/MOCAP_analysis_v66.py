# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 15:54:49 2021

@author: Gilles.DELBECQ

Analysis of MOCAP CSV output

v1 14/10/21
v6 10/03/22
- Added classes for mocap file and the data file
- Will split this script in different analysis ?


Adapted from Ludovic SPAETH

"""




class MOCAP_file:
    
    def __init__(self,filepath):
        import pandas as pd
        self.df_raw = pd.read_csv(filepath,sep=',',header=2,delimiter=None,na_values='') #read the csv output of MOCAP
        self.filepath=filepath #path to the csv file
        self.frequency=100 #Hz
    
    
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
        return str(self.filepath.split('\\')[-1].split('_')[0])
    
    def session_idx(self):
        return self.filepath.split('\\')[-1].split('_')[1]
        
    def trial_idx(self):
        return self.filepath.split('\\')[-1].split('_')[2].split('.')[0]     


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
    
    def stance(self, marker):
        """
        Detect stance of a limb
        
        returns peaks index, x, y, z coord of the peak of the limb, lengths in sec of each stance
        """
        
        from scipy.signal import find_peaks 
        
        peaks, _ = find_peaks(-self.coord(marker)[2],prominence=5)
        
        x,y,z=[],[],[]
        for i in peaks:
            y.append(self.coord(marker)[0][i])
            x.append(self.coord(marker)[1][i])
            z.append(self.coord(marker)[2][i])
        
        lenghts = []
        
        for i in range(len(peaks)):
            
            if i != 0:
                lenghts.append((peaks[i]-peaks[i-1])*1/self.frequency)        
        
        return peaks,x,y,z,lenghts
    
    def step_height(self, marker):
        """
        Detect height of each step
        
        returns peaks index, x, y, z coord of the peak of the limb, lengths in sec of each stance
        """
        
        from scipy.signal import find_peaks 
        
        peaks, _ = find_peaks(self.coord(marker)[2],prominence=5)
        
        x,y,z=[],[],[]
        for i in peaks:
            y.append(self.coord(marker)[0][i])
            x.append(self.coord(marker)[1][i])
            z.append(self.coord(marker)[2][i])

        return peaks,x,y,z

    
    def calculate_angle(self,markerA,markerB,markerC):
        """
        Calculate angle formed in markerB
        https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates
        """
        
        A,B,C=np.asarray(self.coord(markerA)),np.asarray(self.coord(markerB)),np.asarray(self.coord(markerC))
        ba = A - B
        bc = C - B
        
        cosine_angle=[]
        for i in range(len(ba[0])):
            cosine_angle.append(np.dot(ba[:,i], bc[:,i]) / (np.linalg.norm(ba[:,i]) * np.linalg.norm(bc[:,i])))
        
        angle = np.degrees(np.arccos(cosine_angle))
        return angle

    def speed(self,marker):
        
        
        pos=self.coord(marker)
        speed=[]
        
        for i in range(len(pos[0])):
            if i !=0:
                try:
                    speed.append((((pos[0][i]-pos[0][i-1])**2)+((pos[1][i]-pos[1][i-1])**2)+((pos[2][i]-pos[2][i-1])**2))/1000*self.frequency)
                except:
                    pass # doing nothing on exception
        
        return speed
    
    def flatten(self,marker):
        
        import statistics as stat
        
        start = self.coord(f"{self.subject()}:Platform1")
        stop = self.coord(f"{self.subject()}:Platform2")
        
        start_x,start_z=stat.median(start[1]),stat.median(start[2])
        stop_x,stop_z=stat.median(stop[1]),stat.median(stop[2]) 
        
        coef_dir = (stop_z-start_z)/(stop_x-start_x)
        
        x = self.coord(marker)[1]
        y = self.coord(marker)[0]
        
        new_z = []
        
        for i in range(len(self.coord(marker)[1])):
            shift = self.coord(marker)[1][i]*coef_dir
            new_z.append(self.coord(marker)[2][i]-shift)
        
        new_z = np.array(new_z)
        
        new_coords = np.array([y,x,new_z])
        
        return new_coords
 


class DATA_file:
    def __init__(self,filepath):
        import pandas as pd
        self.df_data_info = pd.read_excel(filepath,sep=',',header=0,delimiter=None,na_values='')
        self.filepath=filepath    

    def get_info(self,subject,session,trial):
        
        infos = self.df_data_info.loc[((self.df_data_info['Animal'] == int(subject)) & (self.df_data_info['Session'] == int(session))& (self.df_data_info['Trial'] == int(trial)))]   
        freq = infos['Freq'].tolist()[0]
        power = infos['Power'].tolist()[0]
        tracking_quality = infos['Tracking quality'].tolist()[0]
        obstacle_idx = infos['Obstacle'].tolist()[0]
        
        return freq,power,tracking_quality,obstacle_idx



file_path = 'D:/Working_Dir/MOCAP/Fev2022/Raw_CSV/1110/1110_01_02.csv'
data_MOCAP = MOCAP_file(file_path)

import os
import pandas as pd
from matplotlib import pyplot as plt 
import numpy as np
from scipy.signal import savgol_filter


root_dir='D:\Working_Dir\MOCAP\Fev2022\Raw_CSV'

data_info_path = 'D:/Working_Dir/MOCAP/Fev2022/Data_info.xlsx'
data_info = DATA_file(data_info_path)


#First Loop : loop on all csv files to list them in the list "Files"
Files = []
for r, d, f in os.walk(root_dir):
# r=root, d=directories, f = files
    for filename in f:
        if '.csv' in filename:
            Files.append(os.path.join(r, filename))
            
print('Files to analyze : {}'.format(len(Files)))
    
# Files = ['D:\\Working_Dir\\MOCAP\\Fev2022\\Raw_CSV\\1113\\1113_01_04.csv']


for file in Files:
    data_MOCAP = MOCAP_file(file)
    infos = data_info.get_info(data_MOCAP.subject(),data_MOCAP.session_idx(),data_MOCAP.trial_idx())
    
    """
    Fig 1 normalized
    
    To do
    Light effect : changer de couleur/découper après le passage de l'IR BEAM
    
    """ 
    

    
    figure1 = plt.figure()
    
    norm_left_foot = data_MOCAP.normalized(f"{data_MOCAP.subject()}:Back1", f"{data_MOCAP.subject()}:Left_Foot")
    norm_right_foot = data_MOCAP.normalized(f"{data_MOCAP.subject()}:Back1", f"{data_MOCAP.subject()}:Right_Foot")
    
    # plt.title(f'Normalized feet trajectory {subject}_{session}_{trial}')
    # ax1 = figure.add_subplot(111)
    
    # plt.plot(-norm_left_foot[1],norm_left_foot[2],color='red',label='Left')
    # plt.plot(-norm_right_foot[1],norm_right_foot[2],color='blue',label='Right')
    
    
    
    
    """
    Fig2 : Raw trajectory
    
    To do :
        IR Beam
    """
    import statistics as stat
    
    # left_foot=data_MOCAP.coord(f"{data_MOCAP.subject()}:Left_Foot")
    # right_foot=data_MOCAP.coord(f"{data_MOCAP.subject()}:Right_Foot")
    
    left_foot=data_MOCAP.flatten(f"{data_MOCAP.subject()}:Left_Foot")
    right_foot=data_MOCAP.flatten(f"{data_MOCAP.subject()}:Right_Foot")    
    
    figure2 = plt.figure()
    plt.title(f'Raw Feet trajectory {data_MOCAP.subject()}_{data_MOCAP.session_idx()}_{data_MOCAP.trial_idx()}')
    plt.plot(-left_foot[1],left_foot[2],color='red',label='Left')
    plt.plot(-right_foot[1],right_foot[2],color='blue',label='Right')
    plt.axvline(stat.median(data_MOCAP.coord(f"{data_MOCAP.subject()}:IR Beam1")[1]))
    
    
    stances = data_MOCAP.stance(f"{data_MOCAP.subject()}:Left_Foot")
    
    
    for i in range(len(list(stances[1]))):
    
        try:
            plt.axvspan(-stances[1][i],-stances[1][i+1], color='red',alpha=0.3)
        except:
            pass
    
    stances = data_MOCAP.stance(f"{data_MOCAP.subject()}:Right_Foot")
    for i in range(len(list(stances[1]))):
        try:
            plt.axvspan(-stances[1][i],-stances[1][i+1], color='blue',alpha=0.3)
        except:
            pass
        
        
    # """
    # Fig3 : full leg trajectory
    
    # To do :
    #     Ir Beam
    # """
    
    # left_Hip,left_Knee,left_Ankle=data_MOCAP.coord(f"{data_MOCAP.subject()}:Left_Hip"),data_MOCAP.coord(f"{data_MOCAP.subject()}:Left_Knee"),data_MOCAP.coord(f"{data_MOCAP.subject()}:Left_Ankle")
    
    # figure3 = plt.figure()
    # plt.title(f'Full leg trajectory {data_MOCAP.subject()}_{data_MOCAP.session_idx()}_{data_MOCAP.trial_idx()}')
    
    
    # # ax3 = figure3.add_subplot(111)   
    # # ax3.set_aspect('equal')    
    
    # plt.plot([list(-left_foot[1]),list(-left_Ankle[1])],[list(left_foot[2]),list(left_Ankle[2])],color='deepskyblue')
    # plt.plot([list(-left_Ankle[1]),list(-left_Knee[1])],[list(left_Ankle[2]),list(left_Knee[2])],color='red')
    # plt.plot([list(-left_Knee[1]),list(-left_Hip[1])],[list(left_Knee[2]),list(left_Hip[2])],color='green')
    
    # # figure3.savefig(f'{root_dir}/Figs/{subject}_{session}_{trial}_Fig3_H.svg')
    # # plt.close()
    
    
        
    # """
    # Fig4 : angles left_foot
    # """
    
    # fig, axs = plt.subplots(3,sharex=True)
    # fig.suptitle(f'Angles {data_MOCAP.subject()}_{data_MOCAP.session_idx()}_{data_MOCAP.trial_idx()}')
    
    # axs[0].plot(savgol_filter(data_MOCAP.calculate_angle(f"{data_MOCAP.subject()}:Left_Knee",f"{data_MOCAP.subject()}:Left_Hip",f"{data_MOCAP.subject()}:Back1"),5,2))
    # axs[0].set_title('Hip')
    # axs[1].plot(savgol_filter(data_MOCAP.calculate_angle(f"{data_MOCAP.subject()}:Left_Ankle",f"{data_MOCAP.subject()}:Left_Knee",f"{data_MOCAP.subject()}:Left_Hip"),5,2))
    # axs[1].set_title('Knee')
    # axs[2].plot(savgol_filter(data_MOCAP.calculate_angle(f"{data_MOCAP.subject()}:Left_Foot",f"{data_MOCAP.subject()}:Left_Ankle",f"{data_MOCAP.subject()}:Left_Knee"),5,2))
    # axs[2].set_title('Ankle')
    
