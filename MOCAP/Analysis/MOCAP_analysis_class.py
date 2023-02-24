# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 15:54:49 2021
@author: Gilles.DELBECQ
Marker list, coord, dataframe from Ludovic SPAETH




Analysis of MOCAP CSV output

v1 14/10/21
v6 10/03/22
- Added classes for mocap file and the data file
- Will split this script in different analysis ?

v7 21/03/22
- Class file to import in the different analysis scripts
- 1 class for the MOCAP output csv file (from vicon)
- 1 class for the excel file with info of the trials

09/05/22
- 1 class for the flattened csv
- get_cycle function

v8 30/01/2023
- Updated for cohort 2 files

"""

def Check_Save_Dir(save_path):
    """
    Check if the save folder exists
    If not : creates it
    
    """
    import os
    isExist = os.path.exists(save_path)
    if not isExist:
        os.makedirs(save_path) #Create folder for the experience if it is not already done
    


class MOCAP_file:
    
    def __init__(self,filepath):
        import pandas as pd
        self.df_raw = pd.read_csv(filepath,sep=',',header=2,delimiter=None,na_values='') #read the csv output of MOCAP
        self.filepath=filepath #path to the csv file
        self.frequency=100 #Hz
    
    
    def get_marker_list(self):
        '''
        Gets marker list from MOCAP file 
        
        Returns :
            List of markers
        '''

        #Get markers list
        markers = [x.split(':')[-1] for x in self.df_raw.columns if 'Unnamed' not in x]
                
        #Assert all markers are string formated
        for marker in markers : 
            assert type(marker) == str, 'Markers are not strings'
        
        return markers

    def subject(self):
        """
        Returns the subject ID from the filepath
        """
        return str(self.filepath.split('\\')[-1].split('_')[0])
    
    def session_idx(self):
        """
        Returns the session ID from the filepath
        """
        return self.filepath.split('\\')[-1].split('_')[1]
        
    def trial_idx(self):
        """
        Returns the trial ID from the filepath
        """
        return self.filepath.split('\\')[-1].split('_')[2].split('.')[0]     

    def whole_idx(self):
        """
        Returns the subject, session, trial IDs from the filepath
        """
        return self.subject(),self.session_idx(),self.trial_idx()

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
        import pandas as pd
        
        data = pd.read_csv(self.filepath,sep=',',header=header,delimiter=None,na_values='')
        
        opt_dataframe = pd.DataFrame(data.values,columns=self.new_file_index())
        return opt_dataframe
    
    
    def coord(self,marker,fstart=1,fstop=-1,projection=None,step=1):
        '''
        Returns array with XYZ coordinates for a single marker
        '''
              
        import numpy as np
        
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
        Detect stance of a limb (when foot is the lowest)
        
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
        Detect height of each step (when foot is the highest)
        
        returns peaks index, x, y, z coord of the peak of the limb, lengths in sec of each stance
        """
        
        from scipy.signal import find_peaks 
        
        peaks, _ = find_peaks(self.coord(marker)[2],prominence=5)
        
        x,y,z=[],[],[]
        for i in peaks:
            y.append(self.coord(marker)[0][i])
            x.append(self.coord(marker)[1][i])
            z.append(self.coord(marker)[2][i])
        
        list_of_peaks=[]
        for i in range(len(peaks)):
            list_of_peaks.append([peaks[i],x[i],y[i],z[i]])
        return list_of_peaks


    
    def calculate_angle(self,markerA,markerB,markerC):
        """
        Calculate angle formed in markerB
        https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates
        """
        
        import numpy as np
        
        A,B,C=np.asarray(self.coord(markerA)),np.asarray(self.coord(markerB)),np.asarray(self.coord(markerC))
        ba = A - B
        bc = C - B
        
        cosine_angle=[]
        for i in range(len(ba[0])):
            cosine_angle.append(np.dot(ba[:,i], bc[:,i]) / (np.linalg.norm(ba[:,i]) * np.linalg.norm(bc[:,i])))
        
        angle = np.degrees(np.arccos(cosine_angle))
        return angle

    def speed(self,marker):
        """
        Calculate the speed of a marker
        
        Returns:
            A list of the speed at each frame
        """
        import math
        import numpy as np
        # pos=self.coord(marker)
        # speed=[]
        
        # for i in range(len(pos[0])):
        #     if i !=0:
        #         try:
        #             speed.append(
        #                 (
        #                     (
        #                         (pos[0][i]-pos[0][i-1])**2)
        #                     +((pos[1][i]-pos[1][i-1])**2)
        #                     +((pos[2][i]-pos[2][i-1])**2)
        #                     )/1000*self.frequency)
        #         except:
        #             pass # doing nothing on exception
        
        # return speed
        
        pos=self.coord(marker)
        speed=[]
        for i in range(len(pos[0])):
            if i !=0:
                try:
                    framepre=(pos[0][i-1],pos[1][i-1],pos[2][i-1])
                    framepost=(pos[0][i],pos[1][i],pos[2][i])
                    
                    vector=math.sqrt(sum((np.array(framepost)-np.array(framepre))**2))#in mm
                    speed.append(vector*self.frequency/1000)#in m/s with /1000
                except:
                    pass # doing nothing on exception
        return speed
    
    def flatten(self,marker):
        """
        Parameters
        ----------
        marker : TYPE
            DESCRIPTION.

        Returns
        -------
        new_coords : TYPE
            np.array([y,x,new_z]).

        """
        
        import statistics as stat
        import numpy as np
        
        start = self.coord(f"{self.subject()}:Platform1")
        stop = self.coord(f"{self.subject()}:Platform2")
        
        start_x,start_z=stat.median(start[1]),stat.median(start[2])
        stop_x,stop_z=stat.median(stop[1]),stat.median(stop[2]) 
        
        coef_dir = (stop_z-start_z)/(stop_x-start_x)
        
        # print(coef_dir)
        # if coef_dir >= 0:
        #     print('+')
        # else:
        #     print('-')
        
        x = self.coord(marker)[1]
        y = self.coord(marker)[0]
        
        new_z = []
        
        for i in range(len(self.coord(marker)[1])):
            if coef_dir >= 0:
                shift = self.coord(marker)[1][i]*coef_dir
                new_z.append(self.coord(marker)[2][i]-shift)
                
            else:
                shift = self.coord(marker)[1][i]*coef_dir
                new_z.append(self.coord(marker)[2][i]-shift)
                
        
        new_z = np.array(new_z)
        
        new_coords = np.array([x,y,new_z])
        
        return new_coords
 

        
        

class DATA_file:
    def __init__(self,filepath):
        self.filepath=filepath    

    def get_info(self,subject,session,trial):
        """
        return freq, power,tracking_quality, obstacle_idx
        """
        import pandas as pd
        df_data_info = pd.read_excel(self.filepath,subject)#,sep=',',header=0,delimiter=None,na_values='')
        
        infos = df_data_info.loc[((df_data_info['Session'] == int(session))& (df_data_info['Trial'] == int(trial)))]   
        power = infos['Stimulation'].tolist()[0]
        tracking_quality = infos['Comment'].tolist()[0]
        task = infos['Task'].tolist()[0]
        
        return power,tracking_quality,task


class Flat_CSV:
    def __init__(self,filepath):
        import pandas as pd
        self.df_flat = pd.read_csv(filepath) #read the flat csv
        self.filepath=filepath #path to the flat csv file
        self.frequency=100 #Hz

    def dataframe(self):
        """
        Returns
        -------
        Dataframe

        """
        
        return self.df_flat
    
      

    def coord(self,marker):
        """
        Get coord of a marker
        
        Returns
        3 arrays (x, y, z)
        
        """
        x=self.dataframe().filter(like=f'{marker}_X').iloc[:, 0]
        y=self.dataframe().filter(like=f'{marker}_Y').iloc[:, 0]
        z=self.dataframe().filter(like=f'{marker}_Z').iloc[:, 0]
        return x.to_numpy(),y.to_numpy(),z.to_numpy()
        

    
    def get_cycles(self,marker):
        """
        get the cycles of walk for designated marker (typically left/right foot)
        
        return indexes of cycles end/start
        """
        def peaks(x,y,z):
            from scipy.signal import find_peaks
            peaks=find_peaks(-z,prominence=5)[0]
            return peaks  

        return peaks(self.coord(marker)[0],self.coord(marker)[1],self.coord(marker)[2])
    
    def get_marker_list(self):
        '''
        Gets marker list from MOCAP file 
        
        Returns :
            List of markers
        '''

        #Get markers list
        markers = [x.split(':')[-1] for x in self.df_raw.columns if 'Unnamed' not in x]
                
        #Assert all markers are string formated
        for marker in markers : 
            assert type(marker) == str, 'Markers are not strings'
        
        return markers

    def subject(self):
        """
        Returns the subject ID from the filepath
        """
        return str(self.filepath.split('\\')[-1].split('_')[0])
    
    def session_idx(self):
        """
        Returns the session ID from the filepath
        """
        return self.filepath.split('\\')[-1].split('_')[1]
        
    def trial_idx(self):
        """
        Returns the trial ID from the filepath
        """
        return self.filepath.split('\\')[-1].split('_')[2].split('.')[0]     

    def whole_idx(self):
        """
        Returns the subject, session, trial IDs from the filepath
        """
        return self.subject(),self.session_idx(),self.trial_idx()

