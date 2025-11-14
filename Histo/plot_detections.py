# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 15:15:21 2025

@author: gdelbecq
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Path to your QuPath exported TSV file
detections = "D:/Microscopie/25-10-09 - 193-FUS/Qupath_193_FUS/measurements_detections.tsv"
annotations = "D:/Microscopie/25-10-09 - 193-FUS/Qupath_193_FUS/measurements.tsv"

RF_acronymes = [
    "MY", "MY-sen", "AP", "CN", "CNlam", "CNspg", "DCO", "VCO", "DCN",
    "CU", "GR", "ECU", "NTB", "NTS", "NTSce", "NTSco", "NTSge", "NTSl", "NTSm",
    "SPVC", "SPVI", "SPVO", "SPVOcdm", "SPVOmdmd", "SPVOmdmv", "SPVOrdm", "SPVOvl",
    "Pa5", "z", "MY-mot", "VI", "ACVI", "VII", "ACVII", "EV", "AMB", "AMBd", "AMBv",
    "DMX", "ECO", "GRN", "ICB", "IO", "IRN", "ISN", "LIN", "LRN", "LRNm", "LRNp",
    "MARN", "MDRN", "MDRNd", "MDRNv", "PARN", "PAS", "PGRN", "PGRNd", "PGRNl",
    "PHY", "NIS", "NR", "PRP", "PMR", "PPY", "PPYd", "PPYs", "VNC", "LAV", "MV",
    "SPIV", "SUV", "x", "XII", "y", "INV", "MY-sat", "RM", "RPA", "RO"
]


# excluded = ['root', 'Root', 'grey', 'BS', 'MY', 'HB', 'fiber tracts', 'CB', 'MY-mot', 'CBX', 'VERM', 'py','Annotation (cells of interest)',
#             'Root object (Image)','x', 'y']
# excluded=[]

# selected=[
#     "GRN", "PARN", "IRN", "IO", "MDRNv", "MARN", "MDRNd",
#     "PPY", "PGRNI", "RPA", "PGRNd", "LNRm", "LRNp", "RO", "PAS"
# ]



# Read the TSV file (columns are separated by tabs)
df_detections,df_annotations = pd.read_csv(detections, sep='\t'),pd.read_csv(annotations, sep='\t')

# df_detections = df_detections[~df_detections['Parent'].isin(excluded)]

selected = list(set(df_annotations['Parent'].unique()) & set(RF_acronymes))


df_detections = df_detections[df_detections['Parent'].isin(selected)]


df_counts = df_detections["Parent"].value_counts().reset_index()
df_counts.columns = ["Parent", "Detection count"]

# Set a nice visual style
sns.set(style="whitegrid")

# Create the bar plot
plt.figure(figsize=(10, 6))
sns.barplot(data=df_counts, x="Parent", y="Detection count", palette="viridis")

# Add titles and labels
plt.title("Number of detections per ROI", fontsize=14, weight='bold')
plt.xlabel("Parent ROI", fontsize=12)
plt.ylabel("Detection count", fontsize=12)

# Rotate x-axis labels for readability
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()