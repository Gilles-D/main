import os
import pandas as pd

# Chemin du dossier contenant les fichiers à renommer
chemin_dossier = r"F:\Data\Microscopie\SOD\1227\Tiff"

# Chemin du fichier Excel contenant les correspondances de noms
chemin_fichier_excel = rf'{chemin_dossier}/liste_fichiers.xlsx'

# Lecture des correspondances de noms à partir du fichier Excel
df = pd.read_excel(chemin_fichier_excel, index_col=0, header=0)

# Boucle sur chaque correspondance de nom
for index, row in df.iterrows():
    nom_actuel = index
    nouveau_nom = row[0]
    chemin_actuel = os.path.join(chemin_dossier, nom_actuel)
    chemin_nouveau = f'{os.path.join(chemin_dossier, nouveau_nom)}.tif'
    os.rename(chemin_actuel, chemin_nouveau)
    print(f"Le fichier {nom_actuel} a été renommé en {nouveau_nom}")
