import numpy as np
import matplotlib.pyplot as plt

# Créer des exemples de tableaux (arrays) de temps d'événements et de temps de stimulation
evenements = np.array([2.5, 5.2, 6.8, 7.9, 10.1, 12.4, 15.6, 18.7, 20.3])
evenements = np.random.uniform(0, 20, 100000)
stimulations = np.array([5.0, 10.0, 15.0])

# Définir la fenêtre temporelle
fenetre_avant = 0.1  # 100 ms avant
fenetre_apres = 0.1   # 100 ms après

# Créer une figure
plt.figure(figsize=(10, 6))

# Créer un tableau pour stocker les positions des événements
event_positions = []

# Parcourir les temps de stimulation
for idx, stimulation in enumerate(stimulations):
    # Sélectionner les événements dans la fenêtre autour du temps de stimulation
    evenements_autour = evenements[(evenements >= stimulation - fenetre_avant) & 
                                    (evenements <= stimulation + fenetre_apres)]
    
    # Calculer les positions relatives des événements par rapport à la stimulation
    relative_positions = evenements_autour - stimulation
    
    # Ajouter les positions relatives au tableau
    event_positions.extend(relative_positions)

# Créer un histogramme des réponses
bins = np.arange(-fenetre_avant, fenetre_apres + 0.001, 0.001)  # Pas de 1 ms
plt.hist(event_positions, bins=bins, color='b', alpha=0.7)

# Marquer les temps de stimulation
plt.vlines(0, 0, plt.gca().get_ylim()[1], color='r', label='Stimulation')

# Étiquettes et titre
plt.xlabel('Temps (s) par rapport à la stimulation')
plt.ylabel('Nombre d\'événements')
plt.title('Histogramme des Réponses Autour des Temps de Stimulation')
plt.legend()

# Afficher la figure
plt.show()
