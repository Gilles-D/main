# -*- coding: utf-8 -*-
"""
Spyder Editor
@author: Matilde Cordero-Erausquin

Performs units analysis : 
    - computes Classification based on "manual" parameters that defined optotagged units => OPTO
    - computes Classification of units based on ephys data (spike and frequ, not optotag) and depth of unit => kephys
    - computes Classification of units based on optotag data (=> 2 or 3 classes, kmeans2 and kmeans3)
    - provides GUI for exploration of all the parameters depending on the Classification
    - provides other exploration and correlation plots
    

Inputs : 
    -  units_metrics (provides characteristics of mean action potential for each unit)
    -  optotag_info_fit (provides characteristics of response to opto stimulation)
    -  spiking ( provides mean, max frequ and properties of autocorrelogram)
    
Outputs : units_data = excel file with parameters for each unit

"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# import PySimpleGUI as sg
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#%%Parameters
animal = '0022_28_07'

spikesorting_path = r'D:/ePhy/SI_Data/spikesorting_results/'
sorter_name = "/kilosort3"
data_folder = "/curated/processing_data"

output_xls = '/units_data.xlsx'

#seuil Z-score réponse en 10ms à stim opto
Z = 5
# seuil % success (réponse dans les 10ms à stim opto)
S = 5

#%% Build unit metrics table


um = pd.DataFrame()
 

### 1. Lire les Units metrics de tous les animaux

umfilename = spikesorting_path + animal + sorter_name + data_folder + "/units_metrics.xlsx"
um = pd.read_excel(umfilename, decimal=",")

    #enlever première colonne qui redonne le numéro des unités
um = um.drop(um.columns[0], axis=1)

    #nommer les unités avec Unit
um['Unit'] = um['Unit'].apply(lambda x: 'Unit_' + str(x))

    #utiliser le numéro de waveform comme index
um.set_index("Unit", inplace=True)

    #corriger la profondeur
correct_depth = 825 - um["Unit depth"]
um["Unit depth"] = -correct_depth

### 2. Lire les Optotag infos de tous les animaux    

optofilename = spikesorting_path + animal + sorter_name + data_folder + "/optotag_infos_fit.xlsx"
optoD = pd.read_excel(optofilename, decimal=",")
 
    #utiliser le numéro de waveform comme index
optoD.set_index(optoD.columns[0], inplace=True)

### 3. Ajouter infos stim à um_temp

um = pd.concat([um, optoD], axis=1)
   
### 4. Lire les données d'analyse spiking 

frequfilename = spikesorting_path + animal + sorter_name + data_folder + "/spiking.xlsx"
spiking = pd.read_excel(frequfilename, decimal=',')

    #utiliser le nom des unités comme index
spiking.set_index(spiking.columns[0], inplace=True)

um = pd.concat([um, spiking], axis=1)

### 5. Définition "manuelle" des neurones optotaggés (true/false)

um['OPTO'] = (um['Z-score'] > Z) & (um['% success'] >= S)
#um['OPTO'] = um['OPTO'].astype(int)  # Convert boolean values to 1 and 0

### 6. Ajouter une colonne Animal
 
um.insert(loc = 0,
          column = 'animal',
          value = animal)

print(um)

label = um.index.values

 

#%% Classification f(ephys + profondeur)

df_ephys = um[['Unit depth', 'peak_to_valley', 'peak_trough_ratio', 'half_width', 'repolarization_slope', 'recovery_slope', 'meanF', 'maxF', 'peak_ACG_bias', 'bsl_ACG_bias']]
df_ephys_scaled = StandardScaler().fit_transform(df_ephys)

silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(df_ephys_scaled)
    silhouette_scores.append(silhouette_score(df_ephys_scaled, kmeans.labels_))

best_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Best number of clusters from ephys: {best_n_clusters}")

    # Kmeans avec nb de classes définis par silhouette
    
kephys = KMeans(
   init="random",
   n_clusters=best_n_clusters,
   n_init=10,
   max_iter=300,
   random_state=42
)

kephys.fit(df_ephys_scaled)

# print(kmeans3.inertia_)
# print(kmeans3.n_iter_)

um['kephys'] = kephys.labels_.tolist()

#%%Only manual opto
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = um

# Créer le nuage de points
plt.figure(figsize=(10, 6))
scatter_plot = sns.scatterplot(
    data=data,
    x='Unit position x', 
    y='Unit depth', 
    hue='OPTO', # Groupe par couleur en fonction de la colonne 'OPTO'
    palette=['red', 'blue'], # rouge pour Faux, bleu pour Vrai
    legend='full'
)

# Titre et labels
scatter_plot.set_title('Nuage de points des Unités avec Profondeur vs Position X')
scatter_plot.set_xlabel('Unit Position X')
scatter_plot.set_ylabel('Unit Depth')

# Inverser l'axe y pour avoir la profondeur en ordre décroissant
scatter_plot.invert_yaxis()

# Afficher le graphique
plt.legend(title='OPTO')
plt.show()

# Compter le nombre d'unités optotagged
opto_true_count = um['OPTO'].sum()
print(rf'{opto_true_count} unités optotaggées')






"""



#%% Classification f(opto)

df_stim_scaled = StandardScaler().fit_transform(optoD)

# Create a boolean mask to identify rows with NaN values
mask = np.isnan(df_stim_scaled).any(axis=1)

# Use boolean indexing to remove rows with NaN values
df_stim_scaled = df_stim_scaled[~mask]

silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(df_stim_scaled)
    silhouette_scores.append(silhouette_score(df_stim_scaled, kmeans.labels_))

best_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Best number of clusters from opto: {best_n_clusters}")

    #================= Kmeans 2 classes =================
    
kmeans2 = KMeans(
   init="random",
   n_clusters=2,
   n_init=10,
   max_iter=300,
   random_state=42
)

kmeans2.fit(df_stim_scaled)

# print(kmeans2.inertia_)
# print(kmeans2.n_iter_)

print(kmeans2.labels_)

um['kmeans2'] = np.nan
um.loc[~mask, 'kmeans2'] = kmeans2.labels_.tolist()

    #================= Kmeans 3 classes =================

kmeans3 = KMeans(
   init="random",
   n_clusters=3,
   n_init=10,
   max_iter=300,
   random_state=42
)

kmeans3.fit(df_stim_scaled)

# print(kmeans3.inertia_)
# print(kmeans3.n_iter_)

print(kmeans3.labels_)

um['kmeans3'] = np.nan
um.loc[~mask, 'kmeans3'] = kmeans3.labels_.tolist()



#%% GUI for exploration

# # Create a scatterplot function
# def generate_plot(x_column, y_column, color_column):
#     fig, ax = plt.subplots(figsize=(5, 5))
#     scatter = plt.scatter(um[x_column], um[y_column], c=um[color_column], cmap='viridis')
#     plt.xlabel(x_column)
#     plt.ylabel(y_column)
#     plt.title(f'{x_column} vs. {y_column}')
#     plt.colorbar(scatter, label=color_column)
#     fig.suptitle(f'{animal}', y=1.02)
#     # Show the plot in a new window
#     plt.show()
    
# # Create a scatterplot function for the 2x2 layout
# def generate_2x2_plot(x_column, y_column, color_columns):
#     fig, axs = plt.subplots(2, 2, figsize=(10, 8))

#     for i, color_col in enumerate(color_columns):
#         row, col = i // 2, i % 2
#         scatter = axs[row, col].scatter(um[x_column], um[y_column], c=um[color_col], cmap='viridis')
#         axs[row, col].set_xlabel(x_column)
#         axs[row, col].set_ylabel(y_column)
#         axs[row, col].set_title(f'{x_column} vs. {y_column}')
#         plt.colorbar(scatter, ax=axs[row, col], label=color_col)

#     plt.tight_layout()
#     fig.suptitle(f'{animal}', y=1.02)
#     plt.show()  # Open the 2x2 layout in a new window


# def create_layout():
#     return [
#         [sg.Text("Select X-axis column:"), sg.Combo(values=list(um.columns), key='x_col')],
#         [sg.Text("Select Y-axis column:"), sg.Combo(values=list(um.columns), key='y_col')],
#         [sg.Text("Select Color by column:"), sg.Combo(values=list(um.columns), key='color_col')],
#         [sg.Button("Generate Plot Individual"),sg.Button("Generate All Plots")],
#         [sg.VSeperator()],
#         [sg.Canvas(key='canvas')]
#     ]

# layout = create_layout()
# window = sg.Window("Scatterplot Generator", layout, resizable=True)
# fig, axs = None, None

# while True:
#     event, values = window.read()

#     if event == sg.WINDOW_CLOSED:
#         break
        
#     if event == "Generate Plot Individual":
#         x_col = values['x_col']
#         y_col = values['y_col']
#         color_col = values['color_col']
#         generate_plot(x_col, y_col, color_col)
        

#     if event == "Generate All Plots":
#         x_col = values['x_col']
#         y_col = values['y_col']
#         color_cols = ['OPTO', 'kephys', 'kmeans2', 'kmeans3']
        
#         generate_2x2_plot(x_col, y_col, color_cols)
        
            
# window.close()




#%% Plots

    #### Distribution of all parameters ####
um.hist(figsize=(16,20), xlabelsize=8, ylabelsize=8)


    #### Distribution of all parameters grouped by ephys classes ####

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 20), sharex=False, sharey=False)
axes = axes.ravel()             # array to 1D
cols = um.columns[1:-1]          # create a list of dataframe columns to use
# print(cols)

for col, ax in zip(cols, axes):
    data = um[[col, 'kephys']]  # select the data
    sns.kdeplot(data=data, x=col, hue='kephys', fill=True, ax=ax, gridsize=150)
    ax.set(title=f'Distribution of: {col}', xlabel=None)

fig.suptitle("Distribution of all parameters grouped by ephys classes", fontsize=20, y=1.01)    
fig.tight_layout()
plt.show()


    #### Distribution of all parameters grouped by 2 opto classes ####

um_drop = um.drop(['kmeans2'], axis=1)

fig6, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 20), sharex=False, sharey=False)
axes = axes.ravel()             # array to 1D
cols = um_drop.columns[1:-1]     # create a list of dataframe columns to use

for col, ax in zip(cols, axes):
    data = um[[col, 'kmeans2']]  # select the data
    sns.kdeplot(data=data, x=col, hue='kmeans2', fill=True, ax=ax, gridsize=150)
    ax.set(title=f'Distribution of: {col}', xlabel=None)
    
fig6.suptitle("Distribution of all parameters grouped by 2 OPTO classes", fontsize=20, y=1.01)    
fig6.tight_layout()
plt.show()


    #### Distribution of all parameters grouped by 3 opto classes ####

um_drop = um.drop(['kmeans3'], axis=1)

fig7, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 20), sharex=False, sharey=False)
axes = axes.ravel()             # array to 1D
cols = um_drop.columns[1:-1]     # create a list of dataframe columns to use

for col, ax in zip(cols, axes):
    data = um[[col, 'kmeans3']]  # select the data
    sns.kdeplot(data=data, x=col, hue='kmeans3', fill=True, ax=ax, gridsize=150)
    ax.set(title=f'Distribution of: {col}', xlabel=None)
    
fig7.suptitle("Distribution of all parameters grouped by 3 OPTO classes", fontsize=20, y=1.01)    
fig7.tight_layout()
plt.show()

    #### Pair-wise global visualisation ####
sns.pairplot(um, hue='kmeans3')



######################
###  Correlations  ###
######################

corr_matrix = um.corr()
print(corr_matrix)

plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, cmap='coolwarm', vmax=1.0, vmin=-1.0, linewidths=0.1, annot=True, annot_kws={"size":8}, square=True)

#####################
### Exploration   ###
#####################

moyennes = um.groupby('kmeans3').mean()

print(moyennes)


"""
#%%Saving

um.to_excel(spikesorting_path + animal + sorter_name + data_folder + output_xls)









