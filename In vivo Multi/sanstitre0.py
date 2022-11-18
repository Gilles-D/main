# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 17:57:31 2022

@author: Gilles.DELBECQ
"""
import pandas as pd
import numpy as np

data = pd.read_excel('//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/Analysis/PCA test/test_waveform.xlsx')

data_array = np.array(data)
