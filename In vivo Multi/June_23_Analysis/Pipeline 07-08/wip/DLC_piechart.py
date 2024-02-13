# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 22:16:27 2023

@author: Gil
"""

import pandas as pd
import matplotlib.pyplot as plt

def concatenate_and_plot(file_paths):
    # Initialize an empty DataFrame to hold all the data
    combined_data = pd.DataFrame()

    # Loop through each file path in the list
    for file_path in file_paths:
        # Read the 'Status' column from the current Excel file
        data = pd.read_excel(file_path, usecols=['Status'])
        # Concatenate the data to the combined DataFrame
        combined_data = pd.concat([combined_data, data], ignore_index=True)

    # Count the occurrences of each unique status in the combined data
    status_counts = combined_data['Status'].value_counts()

    # Create a pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Combined Time Spent in Each Behavior (Status)')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Show the pie chart
    plt.show()

# List of Excel file paths
file_paths = [
"D:/Seafile/Seafile/Ma bibliothèque/Rédaction/Manuscrits/Figures/Data fig/DLC/0022_28_07_01DLC.xlsx",
"D:/Seafile/Seafile/Ma bibliothèque/Rédaction/Manuscrits/Figures/Data fig/DLC/0022_31_07_2DLC.xlsx",
"D:/Seafile/Seafile/Ma bibliothèque/Rédaction/Manuscrits/Figures/Data fig/DLC/0023_28_07_01DLC.xlsx",
"D:/Seafile/Seafile/Ma bibliothèque/Rédaction/Manuscrits/Figures/Data fig/DLC/0023_31_07_2DLC.xlsx",
"D:/Seafile/Seafile/Ma bibliothèque/Rédaction/Manuscrits/Figures/Data fig/DLC/0026_01_08_1DLC.xlsx",
"D:/Seafile/Seafile/Ma bibliothèque/Rédaction/Manuscrits/Figures/Data fig/DLC/0026_29_07_01DLC.xlsx"
]

# Call the function with the list of file paths
concatenate_and_plot(file_paths)
