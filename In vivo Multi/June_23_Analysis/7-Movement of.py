# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 14:38:09 2023

@author: MOCAP
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


filepath="D:/Videos/0012/0026_01_08_1DLC_resnet50_OpenfieldJul31shuffle2_200000_filtered.csv"


def get_coord(filepath, bodypart):
    df = pd.read_csv(filepath)
    
    col_x = df.columns[(df.iloc[0] == bodypart) & (df.iloc[1] == "x")]
    col_y = df.columns[(df.iloc[0] == bodypart) & (df.iloc[1] == "y")]
    
    # Sélectionner la colonne correspondante à l'aide de loc
    selected_colx = df.loc[:, col_x[0]].values
    selected_coly = df.loc[:, col_y[0]].values
    
    return selected_colx[2:].astype(float), selected_coly[2:].astype(float)
    
  
    

def calculer_vitesse(x, y, dt):
    dx = np.diff(x)
    dy = np.diff(y)
    vitesse = np.sqrt(dx**2 + dy**2) / dt
    return vitesse

def distance_entre_points(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


#%%

x,y = get_coord(filepath, "Tail_base")

freq=20.833
time_axis =np.array(range(len(x)))/freq


#%%Trajectoire lissée et reduction artefacts


# Ajouter cette condition pour enlever les points entre 0 et 1
indices_a_supprimer = np.where((x >= 0) & (x <= 5))[0]
# x = np.delete(x, indices_a_supprimer)
# y = np.delete(y, indices_a_supprimer)


# Calculer la distance entre les points consécutifs
distances = distance_entre_points(x[:-1], y[:-1], x[1:], y[1:])

# Définir un seuil pour la distance au-delà duquel on considère qu'il y_smooth a un artefact
seuil_distance = 80

# Indiquer les indices des points où la distance dépasse le seuil (ce sont les artefacts)
indices_artefacts_distance = np.where(distances > seuil_distance)[0] + 1

# Indiquer les indices des points où la vitesse dépasse le seuil (ce sont les artefacts, comme précédemment)
dx_smooth = np.diff(x)
dy_smooth = np.diff(y)
vitesse = np.sqrt(dx_smooth**2 + dy_smooth**2)
seuil_vitesse = 80
indices_artefacts_vitesse = np.where(vitesse > seuil_vitesse)[0]

# Fusionner les indices d'artefacts basés sur la vitesse et la distance
indices_artefacts_total = np.union1d(indices_artefacts_distance, indices_artefacts_vitesse)
indices_artefacts_total = np.union1d(indices_artefacts_total, indices_a_supprimer)

# Supprimer les points correspondants aux_smooth indices des artefacts de la trajectoire
x_corrige = np.delete(x, indices_artefacts_total)
y_corrige = np.delete(y, indices_artefacts_total)
time_axis_corrige = np.delete(time_axis, indices_artefacts_total)

# Plot de la trajectoire brute et de la trajectoire corrigée
plt.figure()
plt.plot(x, y, 'o-', label='Trajectoire brute')
plt.plot(x_corrige, y_corrige, 'r-', label='Trajectoire corrigée')
plt.legend()
plt.xlabel('Position x')
plt.ylabel('Position y')
plt.title('Correction des artefacts dans la trajectoire')
plt.grid(True)
plt.show()


#%%Plot vitesse

# Temps entre les points (à définir en fonction de vos données)
dt = 1.0

# Calculer la vitesse au cours du temps
vitesse = calculer_vitesse(x, y, dt)

# Plot de la vitesse au cours du temps
temps = np.arange(len(vitesse)) * dt
plt.plot(temps, vitesse, 'o-')

for i in indices_artefacts_total:
    plt.axvline(i,c='red')

plt.xlabel('Temps')
plt.ylabel('Vitesse')
plt.title('Vitesse au cours du temps')
plt.grid(True)
plt.show()


#%%Heatmap

# Créer une matrice 2D à partir des positions x et y
positions_matrice = np.vstack((x_corrige, y_corrige))

# Augmenter la résolution en définissant plus de bins
bins = 25

# Créer une heatmap en utilisant plt.hist2d avec une résolution plus élevée
heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)

# Créer le plot de la heatmap
plt.imshow(heatmap, cmap='viridis', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
plt.colorbar()
plt.xlabel('Position x')
plt.ylabel('Position y')
plt.title('Heatmap des points y en fonction de x')
plt.show()

#%%Heatmap seaborn
import seaborn as sns
# Créer une heatmap en utilisant Seaborn
heatmap_data, xedges, yedges = np.histogram2d(x, y, bins=100)
heatmap_data = np.ma.masked_equal(heatmap_data, 0)  # Masquer les zéros pour une meilleure visualisation

# Créer le plot de la heatmap avec Seaborn
sns.heatmap(heatmap_data.T, cmap='hot', vmin=1, vmax=np.max(heatmap_data), cbar=True)

plt.xlabel('Position x')
plt.ylabel('Position y')
plt.title('Heatmap des points y en fonction de x')
plt.show()

#%%

# Exemple de données de vitesse (simulées)
time = time_axis_corrige[1:]  # Temps
velocity = calculer_vitesse(x_corrige,y_corrige,dt=1/freq)  # Vitesse (simulée)

# Définir le seuil de vitesse pour distinguer l'immobilité du mouvement
seuil_vitesse = 20

# Identifier les indices où la vitesse est supérieure au seuil (mouvement)
indices_mouvement = np.where(velocity > seuil_vitesse)[0]

# Identifier les indices où la vitesse est inférieure ou égale au seuil (immobilité)
indices_immobilite = np.where(velocity <= seuil_vitesse)[0]

# Créer un plot montrant la vitesse et les périodes d'immobilité et de mouvement
plt.figure(figsize=(10, 6))
plt.plot(time, velocity, label='Vitesse')
plt.plot(time[indices_immobilite], velocity[indices_immobilite], 'ro', label='Immobilite')
plt.plot(time[indices_mouvement], velocity[indices_mouvement], 'go', label='Mouvement')
plt.axhline(y=seuil_vitesse, color='gray', linestyle='--', label='Seuil')
plt.xlabel('Temps')
plt.ylabel('Vitesse')
plt.legend()
plt.title('Décomposition des phases d\'immobilité et de mouvement')
plt.show()

#%%
# Définir la taille de la fenêtre glissante
taille_fenetre = 10

# Créer un masque avec True pour les points au-dessus du seuil et False pour les points en-dessous du seuil
masque_mouvement = velocity > seuil_vitesse

# Utiliser une fonction de convolution pour identifier les périodes de mouvement
convolution = np.convolve(masque_mouvement, np.ones(taille_fenetre), mode='valid')

# Identifier les indices où la convolution est égale à la taille de la fenêtre (c'est-à-dire où les 10 points consécutifs sont au-dessus du seuil)
indices_mouvement = np.where(convolution == taille_fenetre)[0]

# Identifier les indices où la convolution est égale à zéro (c'est-à-dire où les 10 points consécutifs sont en-dessous ou égaux au seuil)
indices_immobilite = np.where(convolution == 0)[0]

# Créer un plot montrant la vitesse et les périodes d'immobilité et de mouvement
plt.figure(figsize=(10, 6))
plt.plot(time, velocity, label='Vitesse')
plt.plot(time[indices_immobilite], velocity[indices_immobilite], 'ro', label='Immobilite')
plt.plot(time[indices_mouvement], velocity[indices_mouvement], 'go', label='Mouvement')
plt.axhline(y=seuil_vitesse, color='gray', linestyle='--', label='Seuil')
plt.xlabel('Temps')
plt.ylabel('Vitesse')
plt.legend()
plt.title('Décomposition des phases d\'immobilité et de mouvement')
plt.show()



# Déterminer les phases d'immobilité sous forme d'un tableau de tuples (début, fin)
phases_immobilite = []

# Initialiser les indices de début et de fin de la phase d'immobilité
debut_phase = indices_immobilite[0]
fin_phase = indices_immobilite[0]

# Parcourir les indices d'immobilité pour regrouper les périodes consécutives en une seule phase
for i in range(1, len(indices_immobilite)):
    if indices_immobilite[i] == indices_immobilite[i-1] + 1:
        # Si l'indice est consécutif à l'indice précédent, il fait toujours partie de la même phase
        fin_phase = indices_immobilite[i]
    else:
        # Sinon, nous avons trouvé la fin de la phase d'immobilité précédente et nous devons enregistrer cette phase
        phases_immobilite.append((debut_phase, fin_phase))
        # Déplacer les indices de début et de fin pour la prochaine phase
        debut_phase = indices_immobilite[i]
        fin_phase = indices_immobilite[i]

# Ajouter la dernière phase d'immobilité au tableau
phases_immobilite.append((debut_phase, fin_phase))

