# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 09:50:00 2023

@author: Gil
"""

import pandas as pd

# Load the excel file
data = pd.read_excel("G:/Data/ePhy/0022_01_08/processing_data/concatenated.xlsx")

# Extract motion capture parameters columns (excluding time_axis and Unnamed: 0)
motion_columns = [col for col in data.columns if not col.startswith("Unit_") and col not in ["time_axis", "Unnamed: 0"]]

# Extract neuron activity columns
unit_columns = [col for col in data.columns if col.startswith("Unit_")]

# Separate the data into independent and dependent variables
X = data[motion_columns]
y = data[unit_columns]



import statsmodels.api as sm

# Fit GLM model for the first neuron (Unit_2)
model = sm.GLM(y['Unit_2'], sm.add_constant(X))
result = model.fit()

# Display summary for the first neuron
result.summary()


# Function to get significant coefficients for each unit
def get_significant_coefficients(unit_column):
    model = sm.GLM(y[unit_column], sm.add_constant(X))
    result = model.fit()
    significant_vars = result.pvalues[result.pvalues < 0.05].index
    coefficients = result.params[significant_vars]
    return coefficients

# Create a dictionary to store results for each unit
results = {}
for unit in unit_columns:
    coefficients = get_significant_coefficients(unit)
    if not coefficients.empty:
        results[unit] = coefficients

# Convert results dictionary to a DataFrame for better presentation
results_df = pd.DataFrame(results).transpose()
