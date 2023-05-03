# -*- coding: utf-8 -*-
"""
Created on Mon May  9 18:13:30 2022

@author: Gilles.DELBECQ
"""


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import os

file=r'F:/Data/Microscopie/SOD/1226/Tiff/mapping/data.xlsx'
folder = os.path.dirname(os.path.dirname(file))

df = pd.read_excel(file,index_col=0)
# df = df.set_index('x')

slicing_axis='coronal'
microscope_10x_scale=0.65*2 #µm/px
fluo_contour=30 #Takes above this % of maximum fluo

if slicing_axis == "coronal":
    df = df.transpose()

max_of_df=df.max().max()

"""
Heat map with raw values
"""
plt.figure()
ax = sns.heatmap(df,cmap="cubehelix")
plt.contour(np.arange(.5, df.shape[1]), np.arange(.5, df.shape[0]), df, levels=[max_of_df*fluo_contour/100], colors='yellow')
plt.gca().invert_xaxis()


plt.show()
# plt.savefig(r'//equipe2-nas1/Gilles.DELBECQ/Data/Microscopie/Histo électrodes/Injections Chr2 retro/0002_full_res/fig.png')



"""
Normalization from max
"""
norm_df = (df / max_of_df) *100

plt.figure()
ax = sns.heatmap(norm_df,cmap="cubehelix")
plt.gca().invert_xaxis()
plt.contour(np.arange(.5, norm_df.shape[1]), np.arange(.5, norm_df.shape[0]), norm_df, levels=[fluo_contour,100], colors='yellow')
#plt.xlim([0,3000])
# plt.ylim([1000,1500])
#plt.gca().invert_yaxis()
plt.show()

plt.savefig(rf'{folder}\heatmap.svg')

"""
Plot contour only, from raw values
"""
plt.figure()
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.contour(np.arange(.5, norm_df.shape[1]), np.arange(.5, norm_df.shape[0]), norm_df, levels=[fluo_contour,100], colors='yellow')

plt.yticks(np.arange(.5, df.shape[0]),df.index.tolist())

plt.axhline(y=0)
plt.axhline(y=3)
plt.axvline(0)
plt.axvline(1000/microscope_10x_scale)

plt.savefig(rf'{folder}\contour.svg')

# # plt.savefig('C:/Users/Gilles.DELBECQ/Desktop/Valeurs_64/fig_contour.svg')


# for i in range(len(df.axes[1])):
#     test = norm_df.iloc[:,[i]].values.tolist()
#     plt.plot(test)
#     plt.legend()
