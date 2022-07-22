# -*- coding: utf-8 -*-
"""
Created on Mon May  9 18:13:30 2022

@author: Gilles.DELBECQ
"""


import pandas as pd
import numpy as np

df = pd.read_excel(r'D:/Working_Dir/Microscopie/2985/Plot/Plot Profile.xlsx',sep=';')
df = df.set_index('x')

import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt

norm_df = (df - df.min()) / (df.max() - df.min())

ax = sns.heatmap(norm_df)

plt.contour(np.arange(.5, df.shape[1]), np.arange(.5, df.shape[0]), df, [60], colors='yellow')
# plt.xlim([0,3000])
plt.ylim([0,4000])
plt.gca().invert_yaxis()
plt.show()

# plt.savefig('C:/Users/Gilles.DELBECQ/Desktop/Valeurs_64/fig.png')

plt.figure()
plt.contour(np.arange(.5, df.shape[1]), np.arange(.5, df.shape[0]), df, [60], colors='yellow')

# plt.gca().invert_yaxis()
# plt.savefig('C:/Users/Gilles.DELBECQ/Desktop/Valeurs_64/fig_contour.svg')