# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:54:58 2023

@author: MOCAP
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_impedance_map(positions, impedance_values):
    """
    Crée une figure avec des cercles représentant chaque site d'électrode à sa position (x, y).
    La couleur du cercle est déterminée par la valeur d'impédance selon une échelle de couleur.

    Parameters:
        positions (list of tuples): Liste de tuples (x, y) des positions des sites d'électrodes.
        impedance_values (list): Liste des valeurs d'impédance correspondantes pour chaque site.

    Returns:
        None
    """
    # Conversion des listes en tableaux numpy pour faciliter les calculs
    positions = np.array(positions)
    impedance_values = np.array(impedance_values)

    # Création de la figure
    fig, ax = plt.subplots()

    # Tracer un cercle pour chaque site à sa position (x, y) avec une couleur basée sur la valeur d'impédance
    for i, (x, y) in enumerate(positions):
        impedance = impedance_values[i]
        color = plt.cm.jet((impedance - impedance_values.min()) / (impedance_values.max() - impedance_values.min()))  # Echelle de couleur basée sur les valeurs d'impédance
        circle = plt.Circle((x, y), radius=5, color=color)  # Ajustez le rayon du cercle selon vos préférences
        ax.add_patch(circle)

    # Fixer le ratio des axes pour que les cercles apparaissent ronds
    ax.set_aspect('equal')

    # Ajustement des limites des axes pour que tous les cercles soient visibles
    ax.set_xlim(min(positions[:, 0]) - 10, max(positions[:, 0]) + 10)
    ax.set_ylim(min(positions[:, 1]) - 10, max(positions[:, 1]) + 10)

    # Ajout d'une échelle de couleur à la figure pour les valeurs d'impédance
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=impedance_values.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=ax)

    # Affichage de la figure
    plt.show()



sites_location=[[0.0, 250.0],
  [0.0, 300.0],
  [0.0, 350.0],
  [0.0, 200.0],
  [0.0, 150.0],
  [0.0, 100.0],
  [0.0, 50.0],
  [0.0, 0.0],
  [43.3, 25.0],
  [43.3, 75.0],
  [43.3, 125.0],
  [43.3, 175.0],
  [43.3, 225.0],
  [43.3, 275.0],
  [43.3, 325.0],
  [43.3, 375.0]]

intan_order = [20,21,22,23,19,18,17,16,15,14,13,12,8,9,10,11]

impedence_df = pd.read_csv("D:/ePhy/Intan_Data/0026/0026_01_08/impe_01_08.csv")
impedence_values = impedence_df.loc[8:23, 'Impedance Magnitude at 1000 Hz (ohms)'].reindex(intan_order)

plot_impedance_map(sites_location,impedence_values)
