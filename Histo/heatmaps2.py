# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 16:12:43 2025

@author: gdelbecq
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob


"""

Paramètres à modifier

"""

csv_folder = r'D:\CoReSpi\Manips\Histo\carto 193\profile_GFP_left'  # dossier contenant tous les CSV
fluo_contour = 30  # % du maximum pour le contour
cmap_heatmap = "plasma"

pixels_per_um = 0.3846 # échelle des images
max_pixel = 1500  # taille longitudinale du rectangle

bregma_slice_index = 34 #index de la coupe correspondant au bregma
step_um = 80       # 80 µm entre 2 coupes consécutives



"""
Lecture et mise en forme du dataframe

"""

# === Options d’inversion ===
if csv_folder.split('_')[-1] == 'left':
    invert_x = True   # Inverser les données sur l’axe des abscisses
else:
    invert_x = False

# === Récupérer tous les CSV ===
csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
print("CSV trouvés :", csv_files)

# === Concaténer tous les CSV ===
df_list = []
for f in csv_files:
    df_temp = pd.read_csv(f)
    df_list.append(df_temp)

df_all = pd.concat(df_list, axis=0, ignore_index=True)

# === Pivot pour obtenir DataFrame 2D : lignes=Images, colonnes=Distance ===
df_pivot = df_all.pivot(index='Image', columns='Distance', values='Intensity')
df_pivot = df_pivot.sort_index()

# === Normalisation en % du maximum global ===
max_val = df_pivot.values.max()
df_norm = (df_pivot / max_val) * 100  # toutes les valeurs en %


# === Inversion des données si demandé ===
if invert_x:
    x_um = (max_pixel - df_norm.columns.values) / pixels_per_um
else:
    x_um = df_norm.columns.values / pixels_per_um

x_mm = x_um / 1000  # conversion en mm
df_norm.columns = x_mm.round(decimals=3)


"""

Figures

"""

# === Heatmap ===
plt.figure(figsize=(12, 6))
sns.heatmap(df_norm, cmap=cmap_heatmap, cbar_kws={'label': 'Intensity (%)'})

# === Contour ===
plt.contour(
    np.arange(.5, df_norm.shape[1]),
    np.arange(.5, df_norm.shape[0]),
    df_norm.values,
    levels=[fluo_contour],
    colors='yellow'
)

# === Axe X arrondi ===
xticks = plt.xticks()[0]
xtick_labels = np.round(df_norm.columns[xticks.astype(int)], 2)  # arrondi à 2 décimales pour mm
plt.xticks(xticks, xtick_labels)

# === Axe Y personnalisé ===
zero_line = bregma_slice_index     # bregma_slice_index = zéro


n_rows = df_norm.shape[0]

# Calcul : chaque ligne a une valeur (index - zero_line) * 80
y_labels_um = -(np.arange(n_rows) - zero_line) * step_um  # en µm

# Conversion optionnelle en mm (si tu veux)
y_labels_mm = y_labels_um / 1000.0

# Positions des ticks : centre de chaque ligne
y_positions = np.arange(n_rows) + 0.5

# Ne garder qu'une tick sur 3
y_positions_sparse = y_positions[::3]
y_labels_sparse = np.round(y_labels_mm, 3)[::3]

plt.yticks(ticks=y_positions_sparse, labels=y_labels_sparse)

plt.xlabel("Distance (mm)")
plt.ylabel("Position sur l'axe (échelle personnalisée)")
plt.title(f"Heatmap normalisée (%) avec contour >{fluo_contour}% et axes modifiés")
plt.tight_layout()

# === Sauvegarde en SVG ===
suffix = ""
if invert_x: suffix += "_invX"

side = csv_folder.split('_')[-1]
fluorophore = csv_folder.split('_')[-2]

out_svg = os.path.join(csv_folder, f"Heatmap_profile_{fluorophore}_{side}{suffix}.svg")

plt.savefig(out_svg, format='svg')
print(f"✅ Figure sauvegardée en SVG : {out_svg}")

plt.show()