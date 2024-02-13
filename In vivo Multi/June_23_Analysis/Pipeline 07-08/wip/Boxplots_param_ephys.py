import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the Excel file
file_path = 'D:/Seafile/Seafile/Ma bibliothèque/Rédaction/Manuscrits/Figures/Data fig/all_neurons_data.xlsx'  # Remplacez par le chemin de votre fichier
data = pd.read_excel(file_path)


# List of parameters to compare
parameters = [
    "Unit depth",
    "peak_to_valley",
    "peak_trough_ratio",
    "half_width",
    "repolarization_slope",
    "recovery_slope",
    "meanF",
    "maxF",
    "peak_ACG_bias",
    "bsl_ACG_bias"
]

# Create a boxplot for each parameter
for parameter in parameters:
    plt.figure(figsize=(10, 6))  # Set the figure size for each boxplot
    sns.boxplot(x='OPTO', y=parameter, data=data, showfliers=True) #x=OPTO ou kmeans3 ou kephysTOT
    plt.title(parameter)
    plt.savefig(f'D:\Seafile\Seafile\Ma bibliothèque\Rédaction\Manuscrits\Figures\Data fig\opto status2/boxplot_{parameter}.png')  # Save the figure as a PNG file
    plt.show()
