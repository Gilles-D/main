import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --- CHARGEMENT DES DONNÉES ---

# Load the data from the three files
data_0022 = pd.read_excel("//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/0022_units_metrics.xlsx")
data_0023 = pd.read_excel("//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0023_01_08/kilosort3/curated/processing_data/0023_units_metrics.xlsx")
data_0026 = pd.read_excel("//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0026_02_08/kilosort3/curated/processing_data/0026_units_metrics.xlsx")

# --- PRÉ-TRAITEMENT DES DONNÉES ---

# Add an identifier column for each dataset
data_0022["Identifier"] = "0022_Unit_" + data_0022["Unit"].astype(str)
data_0023["Identifier"] = "0023_Unit_" + data_0023["Unit"].astype(str)
data_0026["Identifier"] = "0026_Unit_" + data_0026["Unit"].astype(str)

# Combine the three datasets
combined_data = pd.concat([data_0022, data_0023, data_0026], ignore_index=True)

# Drop the original Unit column and the Unnamed column
combined_data.drop(columns=["Unit", "Unnamed: 0"], inplace=True)

# Separate the features from the identifier
features = combined_data.drop(columns=["Identifier"])
identifier = combined_data["Identifier"]

# Normalize the features
scaled_features = StandardScaler().fit_transform(features)

# --- PCA ---

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)
principal_df = pd.DataFrame(data=principal_components, columns=["Principal Component 1", "Principal Component 2"])

# --- VISUALISATION ---

# Visualize the results
plt.figure(figsize=(12, 8))
for i, ident in enumerate(identifier):
    plt.scatter(principal_df.iloc[i, 0], principal_df.iloc[i, 1], label=ident if i % 10 == 0 else "")
    
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("2D PCA of Units' Metrics")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage

# Compute the linkage matrix for hierarchical clustering
linked = linkage(scaled_features, 'ward')

# Plot the dendrogram
plt.figure(figsize=(12, 8))
dendrogram(linked,
           orientation='top',
           labels=identifier.values,
           distance_sort='descending',
           show_leaf_counts=True)
plt.title("Dendrogramme pour la Classification Hiérarchique")
plt.xlabel("Échantillons")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()
