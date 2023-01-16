# -*- coding: utf-8 -*-
"""
Created on Mon May  9 18:13:30 2022

@author: Gilles.DELBECQ
"""


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

df = pd.read_excel(r'//equipe2-nas1/Gilles.DELBECQ/Data/Microscopie/6567/tiff/data.xlsx',index_col=0)
# df = df.set_index('x')

slicing_axis='coronal'
microscope_10x_scale=0.65 #µm/px




"""
Heat map with raw values
"""

if slicing_axis == "coronal":
    plt.figure()
    df = df.transpose()
    ax = sns.heatmap(df,cmap="cubehelix")
    plt.contour(np.arange(.5, df.shape[1]), np.arange(.5, df.shape[0]), df, levels=[520,2000], colors='yellow')
    plt.gca().invert_xaxis()
else:
    ax = sns.heatmap(df,cmap="cubehelix")
    plt.contour(np.arange(.5, df.shape[1]), np.arange(.5, df.shape[0]), df, levels=[520], colors='yellow')
    
#plt.xlim([0,3000])
# plt.ylim([1000,1500])
#plt.gca().invert_yaxis()
plt.show()
# plt.savefig(r'//equipe2-nas1/Gilles.DELBECQ/Data/Microscopie/Histo électrodes/Injections Chr2 retro/0002_full_res/fig.png')



"""
Normalization from max
To correct....
"""

norm_df = (df - df.min()) / (df.max() - df.min()) #Verifier cette norm !
plt.figure()
ax = sns.heatmap(norm_df,cmap="cubehelix")
plt.gca().invert_xaxis()
plt.contour(np.arange(.5, norm_df.shape[1]), np.arange(.5, norm_df.shape[0]), norm_df, levels=[0.42,1], colors='yellow')
#plt.xlim([0,3000])
# plt.ylim([1000,1500])
#plt.gca().invert_yaxis()
plt.show()


"""
Plot contour only, from raw values
"""
plt.figure()
plt.contour(np.arange(.5, df.shape[1]), np.arange(.5, df.shape[0]), df, levels=[520], colors='yellow')

plt.xticks(np.arange(.5, df.shape[1]),df.columns.tolist())

plt.ylim([0, 3000])

plt.gca().invert_yaxis()
plt.savefig(r'\\equipe2-nas1\Gilles.DELBECQ\Data\Microscopie\Histo électrodes\Injections Chr2 retro\0001\contour.svg')

# plt.savefig('C:/Users/Gilles.DELBECQ/Desktop/Valeurs_64/fig_contour.svg')


for i in range(len(df.axes[1])):
    test = norm_df.iloc[:,[i]].values.tolist()
    plt.plot(test)
    plt.legend()
