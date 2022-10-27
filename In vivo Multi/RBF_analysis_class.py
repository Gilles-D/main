# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:03:55 2022

@author: Gilles.DELBECQ
"""

import numpy as np


def load_rbf(rbf_file,number_of_channel=16):
    """
    
    
    """
    signal=np.fromfile(rbf_file)*1000 #in mV
    signal=signal.reshape(int(len(signal)/number_of_channel),-1).transpose()
    
    return signal


def load_stim_file(stim_file):
    """
    
    """
    
    stim_indexes=np.loadtxt(stim_file)
    return stim_indexes

