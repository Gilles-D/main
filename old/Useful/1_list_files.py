# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 11:04:25 2023

@author: Gilles.DELBECQ
"""

import os
import pandas as pd

# Chemin du dossier contenant les fichiers
chemin_dossier = r"F:\Data\Microscopie\SOD\1227\Tiff"

# Liste des fichiers dans le dossier
fichiers = os.listdir(chemin_dossier)

# Création d'un DataFrame avec les noms de fichiers
df = pd.DataFrame({'Nom du fichier': fichiers})

# Nom du fichier Excel à enregistrer
nom_fichier_excel = "liste_fichiers.xlsx"

# Enregistrement du DataFrame dans un fichier Excel
df.to_excel(os.path.join(chemin_dossier, nom_fichier_excel), index=False)
