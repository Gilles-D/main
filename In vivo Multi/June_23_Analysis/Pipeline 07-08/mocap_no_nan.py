# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 09:05:06 2023

@author: Gil
"""

import pandas as pd
import os

def list_recording_files(path):
    """
    List all CSV files containing the specified session in the name
    in the specified directory and its subdirectories.

    Parameters:
        path (str): The directory path to search for CSV files.
        session (str): The session to search for in the file names.

    Returns:
        list: A list of paths to CSV files containing the session in their name.
    """
    import os

    csv_files = []
    for folderpath, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.lower().endswith(".xlsx") and "mocap" in filename:
                csv_files.append(os.path.join(folderpath, filename))

    return csv_files

path = r"G:\Data\ePhy\0022_01_08\processing_data\sync_data"

savepath = r"G:\Data\ePhy\0022_01_08\processing_data\sync_data\no_nan"

files = list_recording_files(path)

for file in files :
    
    # Load the excel file into a pandas DataFrame
    df = pd.read_excel(file)
    
    # Step 1: Identify columns that have non-NaN values
    param_cols = df.columns[df.notna().any()]
    
    # Remove 'time_axis' from the list of parameter columns
    param_cols = param_cols.drop(["Unnamed: 0", "time_axis"])
    
    # For each parameter column, find the first and last non-NaN index
    first_valid_index = df[param_cols].apply(pd.Series.first_valid_index)
    last_valid_index = df[param_cols].apply(pd.Series.last_valid_index)
    
    # Get the overall first and last valid indices
    overall_first_index = first_valid_index.min()
    overall_last_index = last_valid_index.max()
    
    # Filter rows based on these indices
    filtered_df = df.iloc[overall_first_index:overall_last_index+1]
    
    filename = os.path.basename(file).split(".")[0]
    
    filtered_df.to_excel(rf"{savepath}/{filename}_nonan.xlsx")