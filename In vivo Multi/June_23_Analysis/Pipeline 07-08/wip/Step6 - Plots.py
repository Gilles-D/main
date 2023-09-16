# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:27:59 2023

@author: Gilles Delbecq


Plots for the various analysis

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



#%% Read dataframes with infos

session = "0022_01_08"


df_unit_info =pd.read_excel(rf"D:/ePhy/SI_Data/spikesorting_results/{session}/kilosort3/curated/processing_data/units_metrics.xlsx")

df_optotag_info = pd.read_excel(rf'D:/ePhy/SI_Data/spikesorting_results/{session}/kilosort3/curated/processing_data/optotag_infos.xlsx')

df_correlation_speed = pd.read_excel(rf'D:/ePhy/SI_Data/spikesorting_results/{session}/kilosort3/curated/correlation.xlsx',sheet_name='Speed')
df_correlation_obst =pd.read_excel(rf'D:/ePhy/SI_Data/spikesorting_results/{session}/kilosort3/curated/correlation.xlsx',sheet_name='Obstacle')
df_correlation_z = pd.read_excel(rf'D:/ePhy/SI_Data/spikesorting_results/{session}/kilosort3/curated/correlation.xlsx', sheet_name='Back_Z')

unit_list = df_unit_info['Unit']



#%% Plot 1 : Correlation boxplots + dendrograms

    

    
    
#%% Plot 2 : Scatter plot spike parameters by units, with frequency (color) and optotagged identified
#TODO : size of points for frequency


optotagged_neurons = df_optotag_info.loc[df_optotag_info[df_optotag_info['reliability_scores'] > 60].index , 'units']

correlation_mean = np.array(df_correlation_speed.head(14).mean())   #First rows = catwalk
# correlation_mean = np.array(df_correlation_speed.tail(3).mean())  #Bottom rows = obstacle


plt.figure()

optotagged_mask = df_unit_info.index.isin(optotagged_neurons.index)

unit_depth = 825 - df_unit_info['Unit depth']

# Tracer le scatter plot avec des tailles de points basées sur correlation_mean
scatter_plot = sns.scatterplot(
    data=df_unit_info,
    x='peak_to_valley',
    y='repolarization_slope',
    hue=optotagged_mask,
    size=correlation_mean,  # Utilisez correlation_mean pour définir la taille des points
    sizes=(10, 300),  # Personnalisez la plage de tailles des points
)

# Ajouter les annotations
for i, unit in enumerate(unit_list):
    plt.annotate(unit, (df_unit_info['peak_to_valley'][i], df_unit_info['half_width'][i]))

# Afficher le scatter plot
plt.show()


# plt.savefig(rf"{sorter_folder}\curated\waveforms\waveforms_parameters.png")

    
    
#%% Plot 3 : PCA spike_params
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = df_unit_info.iloc[:, 4:]

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

pca = PCA()
principal_components = pca.fit_transform(data_scaled)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='--')
plt.xlabel('Nombre de composantes principales')
plt.ylabel('Variance expliquée cumulée')
plt.show()

# Par exemple, si vous voulez capturer 95% de la variance
n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1

# Appliquez à nouveau la PCA avec le nombre de composantes sélectionné
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(data_scaled)

from scipy.cluster import hierarchy

dendrogram = hierarchy.dendrogram(hierarchy.linkage(principal_components, method='ward'))
plt.title('Dendrogramme')
plt.xlabel('Unités')
plt.ylabel('Distance')
plt.show()
