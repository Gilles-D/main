# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:23:25 2019

This code extracts raw figures and xcel files (markers positions) from Thy1cop4beam experiment

@author: Ludovic.SPAETH
"""

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
    import numpy as np
   
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


def marker_matrix(file,marker,fstart=1,fstop=-1,method='YZ'):
    
    import numpy as np
   
    data = dataframe(file)
    
    if fstop == -1:
        stop = data.shape[0]-1
    else:
        stop = fstop
        
    xs = data.iloc[fstart:stop,data.columns.get_loc('{}:X'.format(marker))].values
    ys = data.iloc[fstart:stop,data.columns.get_loc('{}:Y'.format(marker))].values
    zs = data.iloc[fstart:stop,data.columns.get_loc('{}:Z'.format(marker))].values

    if method == 'YZ':
        matrix = np.vstack((ys,zs))

    else:
        matrix = np.vstack((xs,ys,zs))

    return matrix     


def normed_trajectory(folder,condition='Stim',ref_mark_tag = 'Back1',
                      markers_ref = 'L_Foot2',plot=False) :
    import os 
    
    if plot == True :
        plt.figure()
        plt.title('{}_{}'.format(folder,condition))
        
    X, Y, Z = [],[],[]
    
    for file in os.listdir(folder):
        
        if file.endswith('.csv'):
            
            if 'Cal' not in file : 
                
                if condition in file:     
                   
                    marker_list = get_marker_list('{}/{}'.format(folder,file))
                    
                    ref_marker = str([x for x in marker_list if ref_mark_tag in x][0])
                    
                    print (file)
                    print ('Ref Marker is',ref_marker)
                    
                    ref_x,ref_y,ref_z =coord('{}/{}'.format(folder,file),ref_marker,fstart=1,fstop=-1,projection=None)
                    
                    markers = []
                    for h in markers_ref :
                        markers.append(str([x for x in marker_list if h in x][0]))
                        
                    print (markers)
                        
                        
                    for marker in markers: 

                        x,y,z = coord('{}/{}'.format(folder,file),marker,fstart=1,fstop=-1,projection=None)      
                        
                        X.append(x)
                        Y.append(y-ref_y)
                        Z.append(z)

    return X, Y, Z
                    
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------ 
if __name__ == '__main__': 
    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    import os 
    import pandas as pd 
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    
    #Animals to analyse
    animals = ['B1','B2','C1','C2','C3','C4','C5','D2']
    animals = ['D2']

    
    #General folder to save excel sheets and plots
    save_destination = 'U:/10_MOCAP/ThyOne_Beam_Analysis_from_corected_dataset'
    data_source = 'U:/10_MOCAP/Thy1Beam_CORRECTED'
    
    #Save data and/or figures
    savefig = True
    savedata = True
    
    #Loop for every animal
    for animal in animals: 
        
        print('')
        print ('Animal : {}'.format(animal))
    
        datafolder = '{}/{}'.format(data_source,animal)
        
        savedir = '{}/{}'.format(save_destination,animal)
        
        #Create animal folder in datadir if not present
        #if savedir not in ['{}/{}'.format(save_destination,x) for x in os.listdir(save_destination) if '.enf' not in x]:
        try:
            os.makedirs(savedir)
        except FileExistsError:
            pass
         
        #markers_ref = ['L_Foot2', 'R_Foot2'] #Complete or incomplete name of markers of interest
        #markers_ref = ['Back2', 'Back3'] #Complete or incomplete name of markers of interest
        #colors = ['orange','skyblue']
        
        #For later : normalized trajectories 
        #L_pawx,L_pawy,L_pawz = normed_trajectory(folder,condition='Try',markers_ref = 'L_Foot2')
        #
        #plt.figure()
        #plt.title(folder)
        #for trial in range(len(L_pawx)):
        #    
        #    plt.plot(L_pawy[trial],L_pawz[trial],label=trial)
        #    plt.xlabel('Y(mm)'); plt.ylabel('Z(mm)')
        
        folder_list = ['{}/{}'.format(datafolder,x) for x in os.listdir(datafolder) if '.enf' not in x]
        
        for folder in folder_list:
            
            print ('Session : {}'.format(folder.split('/')[-1]))
            
            save_path = '{}/{}'.format(savedir,folder.split('/')[-1])
            
            try:
                os.makedirs(save_path)
            except FileExistsError:
                pass
                
            file_list = ['{}/{}'.format(folder,x) for x in os.listdir(folder) if '.csv' in x]
            
            for file in file_list:
                
                #Get file name for saving
                file_tag = file.split('/')[-1].split('.')[0]
                print('File : {}'.format(file_tag))
                
                #Raw plots---------------------------------------------------------------------    
                fig = plt.figure(figsize=(10,8))
                ax = fig.add_subplot(111,projection='3d')
                fig.suptitle('{} coordinates'.format(file))
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
                ax.set_zlabel('Z (mm)')
                ax.view_init(elev=16., azim=-60)
                
                fig2, axx = plt.subplots(1,1,figsize=(16,6))
                axx.set_xlabel('position Y (mm)')
                axx.set_ylabel('position Z (mm)')
                axx.set_title('{} flat'.format(file))
                 
                marker_list = get_marker_list(file)
            
                with pd.ExcelWriter('{}/{}_coordinates.xlsx'.format(save_path,file_tag)) as writer:
                
                    for marker in marker_list:
                        
                        if 'zrg' in marker :
                            continue
                    
                        x,y,z = coord(file,marker,fstart=1,fstop=-1,projection=None)
                    
                        ax.scatter(x,y,z,label=marker)
                        axx.scatter(y,z,label=marker)
                        
                        marker_tag = marker.split(':')[1]
                        
                        df = pd.DataFrame(np.vstack((x,y,z)).transpose(),
                                          columns=['{}_X(mm)'.format(marker_tag),'{}_Y(mm)'.format(marker_tag),'{}_Z(mm)'.format(marker_tag)])
                        
                        if savedata == True:
                            try:
                                df.to_excel(writer,sheet_name=marker_tag)
                            except :
                                #Sometines markers are doubled in a same datasheet
                                df.to_excel(writer,sheet_name='{}_BIS'.format(marker_tag))
                                
                        
                    ax.legend(loc='best')
                    axx.legend(loc='best')
                    
                    if savefig == True: 
                        fig.savefig('{}/{}_3D_axes.pdf'.format(save_path,file_tag))
                        plt.close()
                        fig2.savefig('{}/{}_YZ_projection.pdf'.format(save_path,file_tag))
                        plt.close()
                
                    

            














#    for file in os.listdir(folder):
#        
#        if file.endswith('.csv'):
#            
#            if 'Cal' not in file : 
#                
#                if 'Stim' in file: 
#    
#                    plt.title('Stim')
#                   
#                    marker_list = get_marker_list('{}/{}'.format(folder,file))
#                    
#                    
#                    ref_marker = str([x for x in marker_list if ref_mark_tag in x][0])
#                    
#                    print (file)
#                    print ('Ref Marker is',ref_marker)
#                    
#                    ref_x,ref_y,ref_z =coord('{}/{}'.format(folder,file),ref_marker,fstart=1,fstop=-1,projection=None)
#                    
#                    markers = []
#                    for h in markers_ref :
#                        markers.append(str([x for x in marker_list if h in x][0]))
#                        
#                        
#                    for marker, color in zip(markers, colors) : 
#
#                        x,y,z = coord('{}/{}'.format(folder,file),marker,fstart=1,fstop=-1,projection=None)                      
#                    
#                        plt.plot(y-ref_y,z,label=marker,color=color)         


