# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:48:34 2020

@author: Gilles.DELBECQ
"""
import numpy as np 
from matplotlib import pyplot as plt 
import pandas as pd
import os
import math
import seaborn as sn
from progressbar import ProgressBar
from scipy.signal import savgol_filter

import PySimpleGUI as sg

class Analyse:
    
    def __init__(self):
        self.pbar = ProgressBar()
        self.likelihood_value = 0.9
        self.frequency = 120
        
        self.bdy_pt = 22 #bodypart passing time
        self.bdy_tr = 19 #bodypart trajectory
        self.bdy_is = 10 #bodypart instantaneous speed      
        
    def load(self, root_dir):
        """
        Load all DeepLabCut output csv files.

        Parameters
        ----------
        root_dir : Path of the directory containing the DeepLabCut output csv files.

        Returns
        -------
        Files
            List of the DeepLabCut output csv files.
        df_data
            Work dataframe containing the CSV files and their identification.

        """
        self.Files = []
        for r, d, f in os.walk(root_dir):
        # r=root, d=directories, f = files
            for filename in f:
                if '.csv' in filename:
                    self.Files.append(os.path.join(r, filename))
        print('Fichiers à analyser : {}'.format(len(self.Files)))
        
        data_animal = []
        data_session=[]
        data_trial=[]
        data_csv=[]
        
        for File in self.Files:  
            Session = 0 #Reset indexes
                
            df_raw = pd.read_csv(File) #Load file
            
            name = os.path.split(File) # Split name from path to get classification values
            name = name[1].split('.', )
            name = name[0].split()
            
            #Determine session number
            if name[3]=='J1' and name[4]=='12mm' and name[5]=='C':
                Session = 1
            if name[3]=='J1' and name[4]=='10mm' and name[5]=='C':
                Session = 2
            if name[3]=='J2' and name[4]=='10mm' and name[5]=='C':
                Session = 3
            if name[3]=='J2' and name[4]=='10mm' and name[5]=='R':
                Session = 4
            if name[3]=='J3' and name[4]=='10mm' and name[5]=='R':
                Session = 5
            if name[3]=='J4' and name[4]=='10mm' and name[5]=='R':
                Session = 6
            if name[3]=='J4' and name[4]=='8mm' and name[5]=='R':
                Session = 7
            if name[3]=='J5' and name[4]=='8mm' and name[5]=='R':
                Session = 8
            if name[3]=='J5' and name[4]=='6mm' and name[5]=='R':
                Session = 9
            if name[3]=='J6' and name[4]=='6mm' and name[5]=='R':
                Session = 10
            if name[3]=='J7' and name[4]=='6mm' and name[5]=='R':
                Session = 11
            if name[3]=='J8' and name[4]=='6mm' and name[5]=='R':
                Session = 12
            if name[3]=='J9' and name[4]=='6mm' and name[5]=='R':
                Session = 13
            if name[3]=='J10' and name[4]=='6mm' and name[5]=='R':
                Session = 14
            if name[3]=='J11' and name[4]=='6mm' and name[5]=='R':
                Session = 15
            if name[3]=='J12' and name[4]=='6mm' and name[5]=='R':
                Session = 16
            if name[3]=='J13' and name[4]=='6mm' and name[5]=='R':
                Session = 17
            if name[3]=='J14' and name[4]=='6mm' and name[5]=='R':
                Session = 18
            if name[3]=='J15' and name[4]=='6mm' and name[5]=='R':
                Session = 19
            data_animal.append(name[2])
            data_session.append(Session)
            data_trial.append(name[6][0])
            data_csv.append(df_raw)
        self.df_data = pd.DataFrame({'Animal' : data_animal, 'Session' :  data_session, 'Trial' : data_trial, 'csv' : data_csv})
        return self.Files, self.df_data
    
    def calculateDistance(self, x1,y1,x2,y2):  
        """
        Calculate distance between two points

        Parameters
        ----------
        x1,y1 : Coordinates of a point

        x2, y2 : Coordinates of a point

        Returns
        -------
        dist
            Calculated distance.

        """
        self.dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
        return self.dist
    
    def flatten(self, x, y):
        """
        Flatten the the global trajectory.
        Calculate the linear equation formed by the starting and ending points.
        Substract f(x) from y of the point indicated. 

        Parameters
        ----------
        x,y : Coordinates of a point

        Returns
        -------
        New y with f(x) substracted

        """
        m = (self.end_coord[1] - self.start_coord[1]) / (self.end_coord[0] - self.start_coord[0])
        c = (self.end_coord[1] - (m * self.end_coord[0]))
        return  y-(x * m + c)
    
    def do_the_analysis(self, root_dir):
        """
        Perform the analysis of the passing time from the DLC output CSV files.
        Creates an excel file.

        Parameters
        ----------
        root_dir : Path of the directory containing the DeepLabCut output csv files.

        Returns
        -------
        None.

        """
        
        print('Chargement des fichiers\n')
        self.load(root_dir)
        print('Calcul des temps de passage et vitesse instantanée\n')
        self.writer = pd.ExcelWriter('{}/Analysis.xlsx'.format(root_dir), engine='xlsxwriter')
        
        passing_times=[]
        crossing_idx=[]
        
        
        for i in self.pbar(range(len(self.df_data))):
            #Reset indexes
            index = 0
            row = 0
            crossing_start_idx = 0
            crossing_end_idx = 0

            data = self.df_data.iloc[i][3]
            
            #Determine X coords of start and end using median of both whole columns
            starting_mark_x = data.iloc[2:, 1].median()
            ending_mark_x = data.iloc[2:, 4].median()
            
            """ Passing Time """
            # Check when bodypart crosses starting mark
            for index, row in data.iloc[2:].iterrows():
                if float(data.iloc[index,self.bdy_pt]) >= starting_mark_x and float(data.iloc[index, self.bdy_pt+2])>=self.likelihood_value:
                    crossing_start_idx = data.iloc[index, 0]
                    break
        
            # Check when bodypart crosses ending mark
            for index, row in data.iloc[2:].iterrows():
                if float(data.iloc[index,self.bdy_pt]) >= ending_mark_x and float(data.iloc[index, self.bdy_pt+2])>=self.likelihood_value:
                    crossing_end_idx = data.iloc[index, 0]
                    break
            
            #Translates idexes in time using frequency set in parameters   
            passing_times.append((float(crossing_end_idx)-float(crossing_start_idx)) /self.frequency)
            crossing_idx.append([crossing_start_idx,crossing_end_idx])
        self.df_data['Passing_Time']=passing_times
        self.df_data['Crossing_idx']=crossing_idx
        self.df_data = self.df_data.drop(['csv'], axis=1)
        self.df_data['Fichier']=self.Files
        self.df_data.to_excel(self.writer, sheet_name='Analysis')
        
        # #Excel File
        self.writer.save()
        return

    def plot_the_trajectories(self, root_dir):
        """
        Generates trajectories of each DeepLabCut output csv files.
        Saves it in a dedicated folder in root_dir.

        Parameters
        ----------
        root_dir : Path of the directory containing the DeepLabCut output csv files.

        Returns
        -------
        None.

        """
        self.load(root_dir)
        for i in self.pbar(range(len(self.df_data))):
            #Reset indexes
            traj=0
            
            #Read identification
            a = self.df_data.iloc[i][0]
            s = self.df_data.iloc[i][1]
            t = self.df_data.iloc[i][2]
            data = self.df_data.iloc[i][3]
            
            #Determine X coords of start and end using median of both whole columns
            self.start_coord = [data.iloc[2:, 1].median(),data.iloc[2:, 2].median()]
            self.end_coord = [data.iloc[2:, 4].median(),data.iloc[2:, 5].median()]
            
            "Instantaneous speed"       
            IS_index = []
            IS_speed=[]
            IS_idx = []
                
            for w in data.index[2:]:
                if float(data.iloc[w, self.bdy_is+2])>=self.likelihood_value:
                    IS_index.append(w)
            
            for w in range(len(IS_index)):
                if float(w) == 0:
                    pass
                else:              
                    distance = (self.calculateDistance(float(data.iloc[IS_index[w-1], self.bdy_is]),float(data.iloc[IS_index[w-1], self.bdy_is+1]),float(data.iloc[IS_index[w], self.bdy_is]),float(data.iloc[IS_index[w], self.bdy_is+1]))*52)/self.calculateDistance(self.start_coord[0], self.start_coord[1], self.end_coord[0], self.end_coord[1])
                    time = (IS_index[w]-(IS_index[w-1]))*(1/self.frequency)
                    IS_idx.append(data.iloc[w, self.bdy_tr])
                    IS_speed.append(distance/time)    
            
            "Instantaneous speed plotting"
            if not os.path.exists("{}\Trajectories".format(root_dir)):
                os.makedirs("{}\Trajectories".format(root_dir))
            if not os.path.exists("{}\Trajectories\{}".format(root_dir, a)):
                os.makedirs("{}\Trajectories\{}".format(root_dir, a))
            traj, subplot = plt.subplots(2,1)
            subplot[0].set_title("Instantaneous speed {} {} {}".format(a, s, t))
            try:
                subplot[0].plot(savgol_filter(IS_speed,11,3), color='crimson') 
            except:
                subplot[0].plot(IS_speed, color='crimson')
            subplot[0].xaxis.set_visible(False)
            subplot[0].set_ylabel("Speed (cm/s)")
            subplot[0].set_ylim(0,100)
                
            "Trajectory plotting"
            subplot[1].scatter([self.start_coord[0], self.end_coord[0]], [self.flatten(self.start_coord[0],self.start_coord[1]), self.flatten(self.end_coord[0],self.end_coord[1])], color='crimson', marker='|')
            subplot[1].plot([self.start_coord[0], self.end_coord[0]], [self.flatten(self.start_coord[0],self.start_coord[1])+6, self.flatten(self.end_coord[0],self.end_coord[1])+6], color='black', linewidth=1)
            subplot[1].plot([self.start_coord[0], self.end_coord[0]], [self.flatten(self.start_coord[0],self.start_coord[1])-6, self.flatten(self.end_coord[0],self.end_coord[1])-6], color='black', linewidth=1)
            subplot[1].plot([float(data.iloc[w, self.bdy_tr]) for w in data.index[2:] if float(data.iloc[w, self.bdy_tr+2])>=self.likelihood_value],
                  [self.flatten(float(data.iloc[w, self.bdy_tr]),float(data.iloc[w, self.bdy_tr+1])) for w in data.index[2:] if float(data.iloc[w, self.bdy_tr+2])>=self.likelihood_value], color='crimson') 
            
            subplot[1].set_xlabel("X Coord"), subplot[1].set_ylabel("Y Coord"), subplot[1].set_title("Trajectory")
            subplot[1].set_ylim(self.flatten(self.start_coord[0],self.start_coord[1])-50,self.flatten(self.start_coord[0],self.start_coord[1])+50)
            subplot[1].invert_yaxis()
            traj.savefig("{}\Trajectories\{}\{}_{}_{}.png".format(root_dir,a,a,s,t))
            plt.close(traj)

    def plot_learning_curve(self, excel_path):
        """
        Generates learning plots from the excel file of the passing time analysis.
        Saves it in a dedicated folder in root_dir.

        Parameters
        ----------
        excel_path : do_the_analysis excel output containing the passing times.

        Returns
        -------
        None.

        """
        #Lire le fichier excel
        df_excel = pd.read_excel(excel_path)
        Animal = list(dict.fromkeys(df_excel.Animal.tolist()))
        if not os.path.exists("{}\Learning_plots".format(os.path.dirname(excel_path))):
            os.makedirs("{}\Learning_plots".format(os.path.dirname(excel_path)))
        for a in Animal:
            learning_plot = plt.figure()
            sn.swarmplot([df_excel.iloc[i,2] for i in range(len(df_excel.index)) if df_excel.iloc[i,1] == a and df_excel.iloc[i,2] > 9] , [df_excel.iloc[i,4]for i in range(len(df_excel.index)) if df_excel.iloc[i,1] == a and df_excel.iloc[i,2] > 9],color='C3',size=2)
            sn.boxplot([df_excel.iloc[i,2] for i in range(len(df_excel.index)) if df_excel.iloc[i,1] == a and df_excel.iloc[i,2] > 9], [df_excel.iloc[i,4]for i in range(len(df_excel.index)) if df_excel.iloc[i,1] == a and df_excel.iloc[i,2] > 9], palette=sn.color_palette("coolwarm", 9))
            plt.ylim(0,5), plt.title("Beam test learning (Animal {})".format(a)), plt.xlabel("'Session #'"), plt.ylabel("Passing time (s)")
            learning_plot.savefig("{}\Learning_plots\{}.svg".format(os.path.dirname(excel_path),a))

    def plot_learning_curve_mean(self, excel_path):
        
        df_excel = pd.read_excel(excel_path)

        if not os.path.exists("{}\Learning_plots".format(os.path.dirname(excel_path))):
            os.makedirs("{}\Learning_plots".format(os.path.dirname(excel_path)))
        
        fig1 = plt.figure(1)
        learning_plot_mean = sn.pointplot(x="Session", y="Passing_Time", data=df_excel.query('Session > 9'), hue="Animal", dodge=True, palette=sn.color_palette("pastel", 9)).get_figure()
        plt.xlabel('Session #')
        plt.ylabel('Time (s)')
        plt.title('Mean passing time')
        
        fig2 = plt.figure(2)
        learning_plot_mean_global = sn.lineplot(x="Session", y="Passing_Time", data=df_excel.query('Session > 9')).get_figure()  
        plt.xlabel('Session #')
        plt.ylabel('Time (s)')
        plt.title('Combined average passing time')
        learning_plot_mean.savefig("{}\Learning_plots\Mean Learning Plot.svg".format(os.path.dirname(excel_path)))
        learning_plot_mean_global.savefig("{}\Learning_plots\Global Mean Learning Plot.svg".format(os.path.dirname(excel_path)))
        plt.show()

        # g = sn.FacetGrid(data=df_groups)
        # g.map(plt.errorbar, 'Session', 'mean', 'std', fmt='o', elinewidth=1, capsize=5, capthick=1)

Data_Analyser = Analyse()

sg.theme('DarkBlack')	# Add a touch of color


""""Window : Analysis selection"""

while True:
    find = False
    layout = [  [sg.Text('What would you like to do ?')],
              [sg.Button('Perform the analysis'),sg.Button('Plot the trajectories'),sg.Button('Plot the learning curve'), sg.Button('Cancel')] ]
 
    window = sg.Window('Analysis selection', layout)
    
    
    while True:
        event, values = window.read()
        
        """[Cancel] pressed"""
        if event == sg.WIN_CLOSED or event == 'Cancel':
            window.close()
            find = True
            break
        
        # [Perform the analysis] pressed
        elif event == 'Perform the analysis':
            window.close()
            # Set the root_dir path
            layout2 = [[sg.Text('Select the CSV files directory path:'), sg.InputText(), sg.FolderBrowse()],
                        [sg.Button('Ok'), sg.Button('Cancel')]]
            window_insert_rootdir = sg.Window('Perform the analysis', layout2)
            while True:
                event, values = window_insert_rootdir.read()
                if event == 'Ok':
                    root_dir = values[0]
                    Data_Analyser.do_the_analysis(root_dir)                    
                    window_insert_rootdir.close()
                    break
                elif event == sg.WIN_CLOSED or event == 'Cancel':
                    window_insert_rootdir.close()
                    break

        #[Plot the trajectories] pressed
        elif event == 'Plot the trajectories':
            window.close()
            # Set the root_dir path
            layout2 = [[sg.Text('Select the CSV files directory path:'), sg.InputText(), sg.FolderBrowse()],
                        [sg.Button('Ok'), sg.Button('Cancel')]]
            window_insert_rootdir = sg.Window('Plot the trajectories', layout2)
            while True:
                event, values = window_insert_rootdir.read()
                if event == 'Ok':
                    root_dir = values[0]
                    Data_Analyser.plot_the_trajectories(root_dir)
                    window_insert_rootdir.close()
                    break
                elif event == sg.WIN_CLOSED or event == 'Cancel':
                    window_insert_rootdir.close()
                    break
    
        # [Plot the learning curve] pressed
        elif event == 'Plot the learning curve':
            window.close()
            # Set the excel analysis file path
            layout2 = [[sg.Text('Enter the excel analysis file path:'), sg.InputText(), sg.FileBrowse()],
                       [sg.Checkbox('Individual', default=True, key='Individual'), sg.Checkbox('Mean', key='Mean')],
                        [sg.Button('Ok'), sg.Button('Cancel')]]
            window_insert_excel_path = sg.Window('Please insert the path of the analysis excel file', layout2)
            while True:
                event, values = window_insert_excel_path.read()
                if event == 'Ok' :
                    excel_path = values[0]
                    print(values)
                    if values['Individual'] == True : Data_Analyser.plot_learning_curve(excel_path)
                    if values['Mean'] == True : Data_Analyser.plot_learning_curve_mean(excel_path)
                    window_insert_excel_path.close()
                    break
                elif event == sg.WIN_CLOSED or event == 'Cancel':
                    window_insert_excel_path.close()
                    find = True
                    break
    if find:
        break