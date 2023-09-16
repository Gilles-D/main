# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 10:12:28 2023

@author: Gilles.DELBECQ
"""

import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
import matplotlib.pyplot as plt

class ExcelViewer:
    def __init__(self, root):
        self.df = None
        
        # Sélection du fichier
        self.file_button = tk.Button(root, text="Select xlsx file", command=self.load_file)
        self.file_button.pack(pady=20)
        
        # Liste déroulante pour la sélection des colonnes
        self.combo_label = tk.Label(root, text="Select column(s) to plot")
        self.combo_label.pack(pady=10)
        
        self.combo = ttk.Combobox(root, values=[], postcommand=self.update_combo_values, state="readonly")
        self.combo.pack(pady=10)
        
        # Bouton pour tracer
        self.plot_button = tk.Button(root, text="Plot", command=self.plot_data)
        self.plot_button.pack(pady=20)
        
    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
        if file_path:
            self.df = pd.read_excel(file_path, engine="openpyxl")
            self.combo['values'] = self.df.columns.tolist()
    
    def update_combo_values(self):
        if self.df is not None:
            self.combo['values'] = self.df.columns.tolist()

    def plot_data(self):
        if self.df is not None and self.combo.get():
            selected_column = self.combo.get()
            self.df[selected_column].plot()
            plt.title(selected_column)
            plt.show()

root = tk.Tk()
root.title("Excel Viewer")
app = ExcelViewer(root)
root.mainloop()
