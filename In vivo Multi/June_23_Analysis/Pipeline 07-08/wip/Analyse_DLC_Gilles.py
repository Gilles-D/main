import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, mannwhitneyu

# Fonction pour charger des données depuis un fichier Excel
def load_data(filepath):
    return pd.read_excel(filepath)

# Étape 1: Charger les données de comportement et de fréquence instantanée
# -----------------
behavior_data_path = 'chemin/vers/le/fichier/comportement.xlsx'
rates_data_path = 'chemin/vers/le/fichier/frequence.xlsx'

behavior_data = load_data(behavior_data_path)
rates_data = load_data(rates_data_path)

# Détecter les colonnes correspondant aux unités
unit_columns = rates_data.columns[rates_data.columns.str.startswith('Unit')]

# Étape 2: Préparer et fusionner les données pour l'analyse
# -----------------
# Interpoler les données de comportement pour correspondre aux données de fréquence
behavior_data['Time'] = behavior_data['frames'] * 0.02
behavior_data = behavior_data.set_index('Time').reindex(rates_data['Time']).ffill()

# Fusionner les données sur la base de l'index de temps
combined_data = pd.merge(rates_data, behavior_data, left_on='Time', right_index=True)

# Étape 3: Calculer les fréquences moyennes pour chaque comportement
# -----------------
mean_frequencies_by_behavior = combined_data.groupby('Behavior').mean()[unit_columns]

# Étape 4: Normaliser ces fréquences par rapport à la moyenne totale de la session
# -----------------
overall_mean_frequencies = combined_data[unit_columns].mean()
normalized_frequencies_by_behavior = mean_frequencies_by_behavior / overall_mean_frequencies

# Étape 5: Effectuer le test de Shapiro-Wilk pour la normalité
# -----------------
normality_test_results = {unit: shapiro(combined_data[unit].dropna()) for unit in unit_columns}

# Étape 6: Effectuer le test de Mann-Whitney pour la significativité
# -----------------
mannwhitney_test_results = {behavior: {} for behavior in combined_data['Behavior'].unique()}
for behavior in combined_data['Behavior'].unique():
    behavior_group = combined_data[combined_data['Behavior'] == behavior]
    for unit in unit_columns:
        mannwhitney_test_results[behavior][unit] = mannwhitneyu(
            behavior_group[unit].dropna(),
            combined_data[unit].dropna(),
            alternative='two-sided'
        ).pvalue

# Étape 7: Générer les visualisations correspondantes
# -----------------
# Visualisation des fréquences moyennes pour chaque comportement
mean_frequencies_by_behavior.plot(kind='bar')
plt.title('Mean Instantaneous Frequencies for Each Behavior')
plt.xlabel('Behavior')
plt.ylabel('Mean Frequency')
plt.legend(title='Units')
plt.show()

# Visualisation des fréquences moyennes normalisées pour chaque comportement
normalized_frequencies_by_behavior.plot(kind='bar')
plt.title('Normalized Mean Instantaneous Frequencies for Each Behavior')
plt.xlabel('Behavior')
plt.ylabel('Normalized Mean Frequency')
plt.legend(title='Units')
plt.show()
