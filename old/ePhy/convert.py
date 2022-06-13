# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:57:30 2019
@author: lspaeth (modified by flareno)
Created on Mon Nov 12 14:14:18 2018
This class loads HdF5 recordings from MCS acquisition system as matrices of shape ((channel,data))
Allows to load Raw signals
+ associated time vectors
+ associated sampling rates
All in Volts and Seconds 
Hope it will work 
Then all you have to do is to load HdF5IO from eletroPy package; init class with smthg = HdF5IO(filepath)
After that u can load every instance with associated function, they are all described bellow. 
"""

class HdF5IO:
    
    def __init__(self,filepath):
        import h5py as h5
        file_ = h5.File(filepath,'r')
        
        self.file = file_['Data'] #Loads first node 

#----------RAW RECORDINGS---------------------------------------------------------------------------------------------   
    def raw_record(self): #Gets Raw Records as matrix ((channel,data))
        
        raw = self.file['Recording_0']['AnalogStream']['Stream_0']['ChannelData']
        
        import numpy as np 
        raw_record = np.zeros((raw.shape[0],raw.shape[1]))
        raw_conv = float(self.file['Recording_0']['AnalogStream']['Stream_0']['InfoChannel'][0][9]) #Scaling Factor 
        
        for i in range(raw.shape[0]): #Stores data in new matrix 
            raw_record[i,:] = raw[i,:]/raw_conv #From pV to V
    
        return raw_record
    
    def raw_time(self): #Gets time vector for raw records 
        import numpy as np
        raw_tick = int(self.file['Recording_0']['AnalogStream']['Stream_0']['InfoChannel'][0][8])/1000000.0 #exp6 to pass from us to s
        raw_length = len(self.file['Recording_0']['AnalogStream']['Stream_0']['ChannelData'][0])        
        raw_time = np.arange(0,raw_length*raw_tick,raw_tick)        
        return raw_time
        
    def raw_sampling_rate(self): #Gets sampling rate
        
        raw_tick = float(self.file['Recording_0']['AnalogStream']['Stream_0']['InfoChannel'][0][8])/1000000.0
        
        return 1.0/raw_tick #In Hz
    
    

#---------CONVERT H5 to RAW BINARY-----------------------------------------------------------------------------------

def convert_folder(folderpath, newpath, data_type='filt', transpose=True):
    
    import os, re
    import numpy as np
    
    list_dir = os.listdir(folderpath)
#    folderpath = folderpath
#    newpath = newpath
    
    for file in list_dir:
        
        if file.endswith('.h5'):

            print ('Converting ' + file + '...')
            new_path = '%s/%s'%(folderpath,file)
        
            data = HdF5IO(new_path)

            traces = data.raw_record()
#            print(len(traces[0,:]), data.raw_sampling_rate())
            
            
#             #-----Cut the extra points at the end so that the recording is precisely the set length------
#             sigs = np.empty([16, (int(data.raw_sampling_rate())*length)])
            
#             for i in range(len(traces[:,0])):
#                 if len(traces[i,:]) > (int(data.raw_sampling_rate())*length):
#                     sigs[i,:] = traces[i,:][0:(int(data.raw_sampling_rate())*length)]
# #                    print(len(sigs[i,:]))
                    
#                 # elif len(traces[i,:]) == (int(data.raw_sampling_rate())*length):
#                 else:
#                     sigs[i,:] = traces[i,:]
                    
#                 # else:
#                 #     print('Too few points in the channel {}, the recording may be corrupted'.format(i))
            
#             print(sigs.shape)
            
            #time = data.raw_time()
            
            name = re.sub('\.h5$', '', file)
        
            file_save = '%s/%s.rbf'%(newpath,name)
        
            with open(file_save, mode='wb') as file : 
                
                if transpose == True:                
                    traces.transpose().tofile(file,sep='')
                    
                else:
                    traces.tofile(file,sep='')                    
                
            print ('Conversion DONE')
            
        else:
            print (file + ' is not an h5 file, will not be converted')
            
    print ('Whole directory has been converted successfully')
                
                

if __name__ == '__main__':
    
    
    folderpath = r'C:\Users\MOCAP\Documents\Multi Channel Systems\Multi Channel Experimenter\04-02\HDF5'
    newpath = r'C:\Users\MOCAP\Documents\Multi Channel Systems\Multi Channel Experimenter\04-02\HDF5'
    
    
    convert_folder(folderpath, newpath)