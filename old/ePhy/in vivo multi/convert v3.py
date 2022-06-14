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
import matplotlib.pyplot as plt
import numpy as np

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
        raw_conv = float(self.file['Recording_0']['AnalogStream']['Stream_0']['InfoChannel'][0][10]) #Scaling Factor 
        
        for i in range(raw.shape[0]): #Stores data in new matrix 
            raw_record[i,:] = raw[i,:]/raw_conv #From pV to V
    
        return raw_record
    
    def raw_time(self): #Gets time vector for raw records 
        import numpy as np
        raw_tick = int(self.file['Recording_0']['AnalogStream']['Stream_0']['InfoChannel'][0][9])/1000000.0 #exp6 to pass from us to s
        raw_length = len(self.file['Recording_0']['AnalogStream']['Stream_0']['ChannelData'][0])        
        raw_time = np.arange(0,raw_length*raw_tick,raw_tick)        
        return raw_time
        
    def raw_sampling_rate(self): #Gets sampling rate
        
        raw_tick = float(self.file['Recording_0']['AnalogStream']['Stream_0']['InfoChannel'][0][9])/1000000.0
        
        return 1.0/raw_tick #In Hz
    
    

#---------CONVERT H5 to RAW BINARY-----------------------------------------------------------------------------------

def convert_folder(folderpath, newpath, data_type='raw'):
    
    import os, re
    import numpy as np
    
    list_dir = os.listdir(folderpath)
#    folderpath = folderpath
#    newpath = newpath

    concatenated_file=[] 

    for file in list_dir:
        
        if file.endswith('.h5'):

            print ('Converting ' + file + '...')
            new_path = '%s/%s'%(folderpath,file)
            data = HdF5IO(new_path)
            traces = data.raw_record()
            
            concatenated_file.append(traces)          
                
            print ('Conversion DONE')
            
        else:
            print (file + ' is not an h5 file, will not be converted')
      

    return concatenated_file
    # new_path = '%s/'%(folderpath)
    data = HdF5IO(new_path)

    traces = data.raw_record()
    # sampling_rate = int(data.raw_sampling_rate())
    
    # name = re.sub('\.h5$', '', "concatenated")

    # file_save = '%s/%s_%sHz.rbf'%(newpath,name,sampling_rate)

    # with open(file_save, mode='wb') as file : 

    #         traces.tofile(file,sep='')          
    # print ('Whole directory has been converted successfully')
                
                

if __name__ == '__main__':
    
    
    folderpath = r'C:/Users/Gilles.DELBECQ/Desktop/In vivo Février 2022/H5/15-02'
    newpath = r'C:\Users\Gilles.DELBECQ\Desktop\In vivo Février 2022\RBF/15-02'
    
    
    a = convert_folder(folderpath, newpath)
    array_final = np.array([])
    array_final = np.concatenate(a,axis=0)
    file_save = 'C:/Users/Gilles.DELBECQ/Desktop/In vivo Février 2022/H5/15-02/concatenated.rbf'
    with open(file_save, mode='wb') as file : 
        array_final.tofile(file,sep='')
