# -*- coding: utf-8 -*-
"""
v1.6 03/06/2021
- Ability to use trials above 9 (2 digits trial)
- Light status
- Fail and none detection

v1.5 08/10/2020
- Modified sessions method for learning curves (starts from session 1)

v1.4 18/09/2020 - 14/10/2020
- Automated column indexes
- Added task selector
- Commenting
- WIP Ladder analysis

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
from pathlib import Path
import pingouin as pg
from pingouin import pairwise_ttests, read_dataset

import PySimpleGUI as sg

class Analyse:
    
    def __init__(self):
        self.pbar = ProgressBar()
        self.likelihood_value = 0.9
        self.frequency = 197
        
    def load(self, root_dir):
        """
        Load all DeepLabCut output csv files.
            First Loop on all files to list them in the list "Files"
            
            Second Loop on the list "Files" To generate the work dataframe
                For each csv file :
                    read the csv (df_raw)
                    get animal,session,trial number and append it to the data_animal, data_session and data_trial list
                    append the csv dataframe (df_raw) to the list of csv dataframe


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
        self.df_data = 0
        self.Files = []
        
        #First Loop : loop on all csv files to list them in the list "Files"
        for r, d, f in os.walk(root_dir):
        # r=root, d=directories, f = files
            for filename in f:
                if '.csv' in filename:
                    self.Files.append(os.path.join(r, filename))
        print('Files to analyze : {}'.format(len(self.Files)))
        
        #Initialize lists
        data_animal = [] #List the animal number of all csv files
        data_session=[] #List the session number of all csv files
        data_trial=[] #List the trial number of all csv files
        data_csv=[] #List of dataframes from each csv files
        
        
        #Second Loop to append all the csv files in order to generate the work dataframe
        for File in self.Files:  
            df_raw = pd.read_csv(File) #Load csv file as a dataframe
            
            name = os.path.split(File) # Split name from path to get classification values
            name = name[1].split('.', )
            name = name[0].split()

            data_animal.append(name[1]) #append animal number
            data_session.append(int(name[3])) #append session number
            if name[4][1] != 'D':
                data_trial.append(name[4][0:2]) #append trial number
            else:
                data_trial.append(name[4][0]) #append trial number
            data_csv.append(df_raw) #append the csv values as a dataframe
        
        #Generate the work dataframe df_data, from the lists of animal number, session, trial and dataframes from each csv file
        self.df_data = pd.DataFrame({'Animal' : data_animal, 'Session' :  data_session, 'Trial' : data_trial, 'csv' : data_csv})
                
        return self.Files, self.df_data
    
    def calculateDistance(self, x1,y1,x2,y2):  
        """
        Calculate distance between two points
            Used for speed calculation

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
        Flatten (in y axis) the the global trajectory to compensate the diagonal trajectory of the setup.
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
    
    def range1(self, start, end):
        """
        Range modified function to include all the values when used in len() or list()
        """
        return range(start, end+1)
    
    def get_column_indexes(self, data):
        """
        Get column index from the dataframe for each body part and start/end points (based on their name)

        Parameters
        ----------
        data : csv file output from DLC containing coords of body parts

        Returns
        -------
        Index for each body part used in analysis and start/end point.

        """
        
        #Loop on each column to find the index of each bodypart
        for i in range(len(data.iloc[0][:])):
            if data.iloc[0][i] == 'Start' and data.iloc[1][i]=='x':
                self.start_coord = [data.iloc[2:, i].median(),data.iloc[2:, i+1].median()]
            if data.iloc[0][i] == 'Stop' and data.iloc[1][i]=='x':
                self.end_coord = [data.iloc[2:, i].median(),data.iloc[2:, i+1].median()]
                
            if data.iloc[0][i] == 'Tail_base' and data.iloc[1][i]=='x':
                self.bdy_pt = i
                
            if data.iloc[0][i] == 'Hind_limb' or data.iloc[0][i] == 'Foot' and data.iloc[1][i]=='x':
                self.bdy_tr = i
                
            if data.iloc[0][i] == 'Eye' and data.iloc[1][i]=='x':
                self.bdy_is = i
            
            if data.iloc[0][i] == 'Hip' and data.iloc[1][i]=='x':
                self.bdy_thigh = i
            if data.iloc[0][i] == 'Knee' and data.iloc[1][i]=='x':
                self.bdy_knee = i
            if data.iloc[0][i] == 'Ankle' and data.iloc[1][i]=='x':
                self.bdy_ankle = i
            if data.iloc[0][i] == 'Foot' and data.iloc[1][i]=='x':
                self.bdy_foot = i
                
        # return(self.start_coord, self.end_coord, self.bdy_pt, self.bdy_tr, self.bdy_is, self.bdy_thigh, self.bdy_knee, self.bdy_ankle, self.bdy_foot)
    
    def analysis_beam(self, root_dir):
        """
        Perform the analysis of the passing time from the DLC output CSV files.
        Detect Fails and None
            Fail = did not complete the course at all, or in time (max 10sec)
        Setup on/off light : requires manual 
        Creates an excel file.

        Parameters
        ----------
        root_dir : Path of the directory containing the DeepLabCut output csv files.

        Returns
        -------
        None.

        """
        
        print('Loading Files...\n')
        self.load(root_dir)
    
        
        print('Calculating passing times and instantaneous speed...\n')
        root_dir_path = Path(root_dir) #Getting folder path containing csv files
        if not os.path.exists('{}/Analysis'.format(root_dir_path.parent)): #Check if analysis folder exists
            os.makedirs('{}/Analysis'.format(root_dir_path.parent)) #If not, creates it
        self.writer = pd.ExcelWriter('{}/Analysis/Analysis.xlsx'.format(root_dir_path.parent), engine='xlsxwriter') #Generate excel file
        
        passing_times=[] #Initialize list containing passing times
        crossing_idx=[] #Initialize list containing frame indexes when tail_base crosses start and end points
        
        trial_status=[]
        
        
        
        for i in self.pbar(range(len(self.df_data))): #Loop on all rows in the working dataframe df_data
            #Reset indexes
            index = 0
            row = 0
            
            #Set crossing indexes to 0, so that when animal doesn't cross, write 0 instead of NaN 
            crossing_start_idx = 0
            crossing_end_idx = 0

            data = self.df_data.iloc[i][3] #data represents the csv file dataframe, stored in the column 3 of the working dataframe df_data
            self.get_column_indexes(data) #get the column indexes for the csv file
            
            """ Passing Time """
            # Loop on every rows, check when bodypart crosses starting mark
            for index, row in data.iloc[2:].iterrows():
                if float(data.iloc[index,self.bdy_pt]) >= self.start_coord[0] and float(data.iloc[index, self.bdy_pt+2])>=self.likelihood_value: #If the x position of the body part (tail_base) is higher than the start point, and the likelihood value is above the threshold
                    crossing_start_idx = data.iloc[index, 0] #append the index of the row to the crossing start index list
                    break
        
            # Loop on every rows, check when bodypart crosses ending mark
            for index, row in data.iloc[2:].iterrows():
                if float(data.iloc[index,self.bdy_pt]) >= self.end_coord[0] and float(data.iloc[index, self.bdy_pt+2])>=self.likelihood_value:
                    crossing_end_idx = data.iloc[index, 0]
                    break
            
            
            
            
            
            #Translates idexes in time using frequency set in initial parameters   
            passing_times.append((float(crossing_end_idx)-float(crossing_start_idx)) /self.frequency)
            crossing_idx.append([crossing_start_idx,crossing_end_idx])
            

            if (float(crossing_end_idx)-float(crossing_start_idx)) /self.frequency == 0:
                trial_status.append("None")
            else:
                if int(self.df_data.iloc[i][2]) >= 9 and int(self.df_data.iloc[i][1]) >5:
                    trial_status.append("On")
                else:
                    trial_status.append("Off")

            
            """
            if passing time = 0 -> None = excluded
            if passing time < 0 -> Fail = should indicate failure
            """
            
            """
            if trial > 5 and session >= 9 
            append on to light_status
            """
            
            
            
            
        #Creates excel file from work dataframe
        self.df_data['Passing_Time']=passing_times
        self.df_data['Crossing_idx']=crossing_idx
        self.df_data = self.df_data.drop(['csv'], axis=1) #remove the column containing the csv dataframes
        self.df_data['Fichier']=[os.path.split(File)[-1].split('_')[0] for File in self.Files]
        self.df_data['Status']=trial_status
        self.df_data.to_excel(self.writer, sheet_name='Analysis')
        
        self.writer.save()
        os.startfile(root_dir_path.parent)#opens the save folder
        
        
        
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
        root_dir_path = Path(root_dir)
        #Loop on all csv files (=row in the work dataframe df_data)
        for i in self.pbar(range(len(self.df_data))):
            #Reset indexes
            traj=0
            
            #Read identification
            a = self.df_data.iloc[i][0]
            s = self.df_data.iloc[i][1]
            t = self.df_data.iloc[i][2]
            data = self.df_data.iloc[i][3]
            self.get_column_indexes(data)
            
            "Instantaneous speed"       
            #Initialize lists of index, speed 
            IS_index = [] #Index of points above likelihood threshold
            IS_speed=[]
            # IS_idx = []
                
            #Loop on all points, Get indexes of points with likelihood above the threshold
            for w in data.index[2:]:
                if float(data.iloc[w, self.bdy_is+2])>=self.likelihood_value:
                    IS_index.append(w)
            
            #Get speed of each point (above the threshold)
            for w in range(len(IS_index)):
                if float(w) == 0: #skip the first value
                    pass
                else:  
                    #calculate the distance between point index w and w-1, then calculate time between them, and then calculate speed and append it
                    distance = (self.calculateDistance(float(data.iloc[IS_index[w-1], self.bdy_is]),float(data.iloc[IS_index[w-1], self.bdy_is+1]),float(data.iloc[IS_index[w], self.bdy_is]),float(data.iloc[IS_index[w], self.bdy_is+1]))*52)/self.calculateDistance(self.start_coord[0], self.start_coord[1], self.end_coord[0], self.end_coord[1])
                    time = (IS_index[w]-(IS_index[w-1]))*(1/self.frequency)
                    # IS_idx.append(data.iloc[w, self.bdy_tr])
                    IS_speed.append(distance/time)    
            
            "Instantaneous speed plotting"
            #Creates trajectories folder + animal folder
            if not os.path.exists("{}\Analysis\Trajectories".format(root_dir_path.parent)):
                os.makedirs("{}\Analysis\Trajectories".format(root_dir_path.parent))
            if not os.path.exists("{}\Analysis\Trajectories\{}".format(root_dir_path.parent, a)):
                os.makedirs("{}\Analysis\Trajectories\{}".format(root_dir_path.parent, a))
            
            #Subplot for the instantaneous speed
            traj, subplot = plt.subplots(2,1)
            subplot[0].set_title("Instantaneous speed {} {} {}".format(a, s, t))
            try:
                subplot[0].plot(savgol_filter(IS_speed,11,3), color='crimson') #If possible (not blank) apply savgol filter
            except:
                subplot[0].plot(IS_speed, color='crimson')
            subplot[0].xaxis.set_visible(False)
            subplot[0].set_ylabel("Speed (cm/s)")
            subplot[0].set_ylim(0,100)
                
            "Trajectory plotting"
            #Plot the starting and ending points
            subplot[1].scatter([self.start_coord[0], self.end_coord[0]], [self.flatten(self.start_coord[0],self.start_coord[1]), self.flatten(self.end_coord[0],self.end_coord[1])], color='crimson', marker='|')
            #plot the two lines showing the beam
            subplot[1].plot([self.start_coord[0], self.end_coord[0]], [self.flatten(self.start_coord[0],self.start_coord[1])+6, self.flatten(self.end_coord[0],self.end_coord[1])+6], color='black', linewidth=1)
            subplot[1].plot([self.start_coord[0], self.end_coord[0]], [self.flatten(self.start_coord[0],self.start_coord[1])-6, self.flatten(self.end_coord[0],self.end_coord[1])-6], color='black', linewidth=1)
            
            #plot the points of the trajectory if they are above likelihood threshold
            subplot[1].plot([float(data.iloc[w, self.bdy_tr]) for w in data.index[2:] if float(data.iloc[w, self.bdy_tr+2])>=self.likelihood_value],
                  [self.flatten(float(data.iloc[w, self.bdy_tr]),float(data.iloc[w, self.bdy_tr+1])) for w in data.index[2:] if float(data.iloc[w, self.bdy_tr+2])>=self.likelihood_value], color='crimson') 
            
            subplot[1].set_xlabel("X Coord"), subplot[1].set_ylabel("Y Coord"), subplot[1].set_title("Trajectory")
            subplot[1].set_ylim(self.flatten(self.start_coord[0],self.start_coord[1])-50,self.flatten(self.start_coord[0],self.start_coord[1])+50)
            subplot[1].invert_yaxis()
            traj.savefig("{}\Analysis\Trajectories\{}\{}_{}_{}.pdf".format(root_dir_path.parent,a,a,s,t))
            plt.close(traj)
        os.startfile(root_dir_path.parent)

    def plot_learning_curve(self, excel_path):
        """
        Generates learning plots from the excel file of the passing time analysis (output from analysis_beam()).
            Exclude light on , None and fails
        Saves it in a dedicated folder in root_dir.

        Parameters
        ----------
        excel_path : analysis_beam excel output containing the passing times.

        Returns
        -------
        """
        #Read excel file output of analysis
        df_excel = pd.read_excel(excel_path)
        Animal = list(dict.fromkeys(df_excel.Animal.tolist())) #List of all animals
        if not os.path.exists("{}\Learning_plots".format(os.path.dirname(excel_path))): #Creates learning plots folder if not existing
            os.makedirs("{}\Learning_plots".format(os.path.dirname(excel_path)))
            
        #Loop on all animal
        for a in Animal:
            #Swarm + boxplot of the passing times for each session
            #With status Off (exclude light on, None, and fails)
            # X axis = session (no skipping) , Y axis = Passing time
            #barplot of fails for each session (with light off)
 
            sessions = df_excel.Session.unique()
            subset_df = df_excel[df_excel["Animal"] == a]
            fails=[]
            fails_session=[]
            
            for i in sessions:
                fails.append(int(subset_df.Status[(subset_df.Passing_Time < 0)&(subset_df.Session == i)&(subset_df.Status == 'Off')].count()))
                fails_session.append(i)
            
            learning_plot, subplot = plt.subplots(2,1,sharex=True) #subplots (lignes,colonnes,sharex=true pour partager le même axe x entre les plots)
            learning_plot.suptitle("Beam test learning (Animal {})".format(a))
            subplot[0].set_ylabel("Passing time (s)")
            sn.swarmplot([df_excel.iloc[i,2] for i in range(len(df_excel.index)) if df_excel.iloc[i,1] == a and df_excel.iloc[i,7] =='Off'and df_excel.iloc[i,4]>0], [df_excel.iloc[i,4]for i in range(len(df_excel.index)) if df_excel.iloc[i,1] == a and df_excel.iloc[i,7] =='Off'and df_excel.iloc[i,4]>0],color='C3',size=2,ax=subplot[0])
            sn.boxplot([df_excel.iloc[i,2] for i in range(len(df_excel.index)) if df_excel.iloc[i,1] == a and df_excel.iloc[i,7] =='Off'and df_excel.iloc[i,4]>0], [df_excel.iloc[i,4]for i in range(len(df_excel.index)) if df_excel.iloc[i,1] == a and df_excel.iloc[i,7] =='Off'and df_excel.iloc[i,4]>0], palette=sn.color_palette("pastel", 15),ax=subplot[0])
            subplot[0].set_ylim(-0,10)
            subplot[1].set_xlabel("Session #"), subplot[1].set_ylabel("Number of fails")
            subplot[1].set_ylim(0,10)
            sn.barplot(ax=subplot[1],x=fails_session, y=fails, palette="pastel")
            
            learning_plot.savefig("{}\Learning_plots\{}.svg".format(os.path.dirname(excel_path),a)) #Save figure
        
        """
        Add sheet with mean values ?
        """

    def plot_learning_curve_mean(self, excel_path):
        """
        Generates learning plots from the excel file of the mean passing time analysis (output from analysis_beam()).
        Saves it in a dedicated folder in root_dir.

        Parameters
        ----------
        excel_path : analysis_beam excel output containing the passing times.

        Returns
        -------
        None.

        """
        #Read excel file output of analysis
        df_excel = pd.read_excel(excel_path)

        if not os.path.exists("{}\Learning_plots".format(os.path.dirname(excel_path))):#Creates learning plots folder if not existing
            os.makedirs("{}\Learning_plots".format(os.path.dirname(excel_path)))
        
        fig2 = plt.figure()
        sn.pointplot(x="Session", y="Passing_Time", data=df_excel.query("Passing_Time > 0 & Status == 'Off'"), hue="Animal", dodge=True, palette=sn.color_palette("pastel", 9)).get_figure()
        #Exclude passing times <0 and status = on
        
        plt.xlabel('Session #')
        plt.ylabel('Time (s)')
        plt.title('Mean passing time')
        plt.show()
        fig2.savefig("{}\Learning_plots\Mean Learning Plot.pdf".format(os.path.dirname(excel_path)))
        
        fig3 = plt.figure()
        sn.lineplot(x="Session", y="Passing_Time", data=df_excel.query("Passing_Time > 0 & Status == 'Off'")).get_figure()  
        #Exclude passing times <0 and status = on
        
        plt.xlabel('Session #')
        plt.xticks(list(set(df_excel['Session'].tolist())))
        plt.ylabel('Time (s)')
        plt.title('Combined average passing time')
        
        fig3.savefig("{}\Learning_plots\Global Mean Learning Plot.pdf".format(os.path.dirname(excel_path)))
        plt.show()


    def stats_effect_weeks(self, excel_path):
        """
        Perform RM ANOVA and pairwise T Test (Holm sidak) on the mean of each week of training for each animal

        Parameters
        ----------
        excel_path : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        df_excel = pd.read_excel(excel_path) #read excel file output from analysis()
        # Classify sessions in weeks
        Week1= list(self.range1(1,9))
        Week2= list(self.range1(10,14))
        Week3= list(self.range1(15,19))
        Week4= list(self.range1(20,24))
        Week5= list(self.range1(25,29))
        week=[]
        
        for i in range(len(df_excel.index)):
            week.append(1 if df_excel.iloc[i,2] in Week1 else 2 if df_excel.iloc[i,2] in Week2 else 3 if df_excel.iloc[i,2] in Week3 else 4 if df_excel.iloc[i,2] in Week4 else 5 if df_excel.iloc[i,2] in Week5 else 'Error')
        
        #Add a column week
        df_excel['Semaine'] = week
        #Group in a new dataframe by animal and session and calculate the mean
        df_stats = df_excel[['Animal','Passing_Time','Semaine']].groupby(['Animal','Semaine']).mean().reset_index()
        
        
        # sn.lineplot(x="Semaine", y="Passing_Time", data=df_stats.query('Semaine > 1'), hue='Animal').get_figure()
        
        #Rearrange in a new dataframe with a column for each week mean
        df_stats_arranged = pd.DataFrame(columns=['Animal', 'Semaine 1', 'Semaine 2', 'Semaine 3', 'Semaine 4', 'Semaine 5'])
        
        Animal = list(dict.fromkeys(df_excel.Animal.tolist()))
        #Loop on every animals to append each animal in the new arranged dataframe
        for a in Animal:
            for i in range(len(df_stats.index)):
                if df_stats.iloc[i,1] == 1 and df_stats.iloc[i,0] == a:
                    df_stats_arranged = df_stats_arranged.append({'Animal': a, 'Semaine 1': df_stats.iloc[i,2], 'Semaine 2': df_stats.iloc[i+1,2], 'Semaine 3': df_stats.iloc[i+2,2], 'Semaine 4': df_stats.iloc[i+3,2], 'Semaine 5' : df_stats.iloc[i+4,2]}, ignore_index=True)
        
        #create a dataframe with a repeated mesure anova
        df_result = pd.DataFrame(pg.rm_anova(dv='Passing_Time', within='Semaine', subject='Animal', data=df_stats, detailed=True))
        #create a dataframe with pairwise t test Holm sidak
        df_post_hocs = pd.DataFrame(pairwise_ttests(dv='Passing_Time', within='Semaine', subject='Animal', data=df_stats, padjust='holm'))
        
        #Save in an excel file containing different sheets
        self.writer = pd.ExcelWriter('{}/Stats.xlsx'.format(Path(excel_path).parent), engine='xlsxwriter')
        df_stats_arranged.to_excel(self.writer, sheet_name='Data')
        df_result.to_excel(self.writer, sheet_name='ANOVA')
        df_post_hocs.to_excel(self.writer, sheet_name='Post Hoc')
        self.writer.save()


    def plot_the_trajectories_ladder(self, root_dir):
        """
        WORK IN PROGRESS
        
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
        root_dir_path = Path(root_dir)
        
        
        
        for i in self.pbar(range(len(self.df_data))):
            #Reset indexes
            traj=0
            
            #Read identification
            a = self.df_data.iloc[i][0]
            s = self.df_data.iloc[i][1]
            t = self.df_data.iloc[i][2]
            data = self.df_data.iloc[i][3]
            self.get_column_indexes(data)
            
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
                    IS_idx.append(data.iloc[w, self.bdy_foot])
                    IS_speed.append(distance/time)    
            
            "Instantaneous speed plotting"
            if not os.path.exists("{}\Analysis\Trajectories".format(root_dir_path.parent)):
                os.makedirs("{}\Analysis\Trajectories".format(root_dir_path.parent))
            if not os.path.exists("{}\Analysis\Trajectories\{}".format(root_dir_path.parent, a)):
                os.makedirs("{}\Analysis\Trajectories\{}".format(root_dir_path.parent, a))
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
            # subplot[1].scatter([self.start_coord[0], self.end_coord[0]], [self.flatten(self.start_coord[0],self.start_coord[1]), self.flatten(self.end_coord[0],self.end_coord[1])], color='crimson', marker='|')
            # subplot[1].plot([self.start_coord[0], self.end_coord[0]], [self.flatten(self.start_coord[0],self.start_coord[1])+6, self.flatten(self.end_coord[0],self.end_coord[1])+6], color='black', linewidth=1)
            # subplot[1].plot([self.start_coord[0], self.end_coord[0]], [self.flatten(self.start_coord[0],self.start_coord[1])-6, self.flatten(self.end_coord[0],self.end_coord[1])-6], color='black', linewidth=1)
            X_list=[]
            Y_list=[]
            for w in data.index[2:]:
                threshold_list = [float(data.iloc[w, self.bdy_foot+2]), float(data.iloc[w, self.bdy_ankle+2]), float(data.iloc[w, self.bdy_thigh+2])]
                if all(i >= self.likelihood_value for i in threshold_list):
                    # x_list = [data.iloc[w, self.bdy_foot], data.iloc[w, self.bdy_ankle], data.iloc[w, self.bdy_knee], data.iloc[w, self.bdy_thigh]]
                    # y_list = [data.iloc[w, self.bdy_foot+1], data.iloc[w, self.bdy_ankle+1], data.iloc[w, self.bdy_knee+1], data.iloc[w, self.bdy_thigh+1]]
                    # subplot[1].scatter(x_list, y_list)
                    X_list.append(data.iloc[w, self.bdy_foot])
                    X_list.append(data.iloc[w, self.bdy_ankle])
                    X_list.append(data.iloc[w, self.bdy_thigh])
                    Y_list.append(data.iloc[w, self.bdy_foot+1])
                    Y_list.append(data.iloc[w, self.bdy_ankle+1])
                    Y_list.append(data.iloc[w, self.bdy_thigh+1])
                    
            subplot[1].plot([float(data.iloc[w, self.bdy_foot]) for w in data.index[2:] if float(data.iloc[w, self.bdy_foot+2])>=self.likelihood_value],[self.flatten(float(data.iloc[w, self.bdy_foot]),float(data.iloc[w, self.bdy_foot+1])) for w in data.index[2:] if float(data.iloc[w, self.bdy_foot+2])>=self.likelihood_value], color='crimson') 
            self.writer = pd.ExcelWriter('{}/Analysis/test{}_{}_{}_{}.xlsx'.format(root_dir_path.parent,a,s,t,self.likelihood_value), engine='xlsxwriter')
            df_test = pd.DataFrame(list(zip(X_list, Y_list)))
            df_test.to_excel(self.writer, sheet_name='test')
            self.writer.save()
            # subplot[1].scatter([float(data.iloc[w, self.bdy_foot]) for w in data.index[2:] if float(data.iloc[w, self.bdy_foot+2])>=self.likelihood_value],
            #       [self.flatten(float(data.iloc[w, self.bdy_foot]),float(data.iloc[w, self.bdy_foot+1])) for w in data.index[2:] if float(data.iloc[w, self.bdy_foot+2])>=self.likelihood_value], color='crimson') 
            # subplot[1].scatter([float(data.iloc[w, self.bdy_ankle]) for w in data.index[2:] if float(data.iloc[w, self.bdy_ankle+2])>=self.likelihood_value],
            #       [self.flatten(float(data.iloc[w, self.bdy_ankle]),float(data.iloc[w, self.bdy_ankle+1])) for w in data.index[2:] if float(data.iloc[w, self.bdy_ankle+2])>=self.likelihood_value], color='chocolate') 
            # subplot[1].scatter([float(data.iloc[w, self.bdy_knee]) for w in data.index[2:] if float(data.iloc[w, self.bdy_knee+2])>=self.likelihood_value],
            #       [self.flatten(float(data.iloc[w, self.bdy_knee]),float(data.iloc[w, self.bdy_knee+1])) for w in data.index[2:] if float(data.iloc[w, self.bdy_knee+2])>=self.likelihood_value], color='chartreuse') 
            # subplot[1].scatter([float(data.iloc[w, self.bdy_thigh]) for w in data.index[2:] if float(data.iloc[w, self.bdy_thigh+2])>=self.likelihood_value],
            #       [self.flatten(float(data.iloc[w, self.bdy_thigh]),float(data.iloc[w, self.bdy_thigh+1])) for w in data.index[2:] if float(data.iloc[w, self.bdy_thigh+2])>=self.likelihood_value], color='turquoise') 

            # subplot[1].set_xlabel("X Coord"), subplot[1].set_ylabel("Y Coord"), subplot[1].set_title("Trajectory")
            # subplot[1].set_ylim(self.flatten(self.start_coord[0],self.start_coord[1])-300,self.flatten(self.start_coord[0],self.start_coord[1])+5)
            subplot[1].invert_yaxis()
            traj.savefig("{}\Analysis\Trajectories\{}\{}_{}_{}.png".format(root_dir_path.parent,a,a,s,t))
            # plt.close(traj)
        os.startfile(root_dir_path.parent)

Data_Analyser = Analyse()

sg.theme('LightGrey5')	# Add a touch of color


""""Window : Analysis selection"""

while True:
    find = False
    layout = [  
        [sg.Text('What would you like to do ?')],
        [sg.Frame(layout=[      
                [sg.Radio('Beam Test', "RADIO1", default=True, size=(10,1), key='Beam Test'), sg.Radio('Ladder Test', "RADIO1",key='Ladder Test')]],
                title='Select the task', relief=sg.RELIEF_SUNKEN)],
        [sg.Button('Perform the analysis'),sg.Button('Plot the trajectories'),sg.Button('Plot the learning curve'), sg.Button('Stat analysis'), sg.Button('Quit')] 
        ]


    window = sg.Window('Analysis selection', layout)
    
    
    while True:
        event, values = window.read()
        
        """[Quit] pressed"""
        if event == sg.WIN_CLOSED or event == 'Quit':
            window.close()
            find = True
            break
        
        # [Perform the analysis] pressed
        # Beam Test
        elif event == 'Perform the analysis':
            layout2 = [[sg.Text('Select the CSV files directory path:'), sg.InputText(), sg.FolderBrowse()],
                        [sg.Button('Ok'), sg.Button('Cancel')]]
            window_insert_rootdir = sg.Window('Perform the analysis - Beam Test', layout2)
            while True:
                event, values0 = window_insert_rootdir.read()
                if event == 'Ok':
                    if values['Beam Test']==True:
                        root_dir = values0[0]
                        Data_Analyser.analysis_beam(root_dir)                    
                        window_insert_rootdir.close()
                    elif values['Ladder Test']==True:
                        root_dir = values0[0]
                        Data_Analyser.analysis_beam(root_dir)                    
                        window_insert_rootdir.close()
                    break
                elif event == sg.WIN_CLOSED or event == 'Cancel':
                    window_insert_rootdir.close()
                    break


        #[Plot the trajectories] pressed
        elif event == 'Plot the trajectories':
            # Set the root_dir path
            layout2 = [[sg.Text('Select the CSV files directory path:'), sg.InputText(), sg.FolderBrowse()],
                        [sg.Button('Ok'), sg.Button('Cancel')]]
            window_insert_rootdir = sg.Window('Plot the trajectories', layout2)
            while True:
                event, values0 = window_insert_rootdir.read()
                if event == 'Ok':
                    if values['Beam Test']==True:
                        root_dir = values0[0]
                        Data_Analyser.plot_the_trajectories(root_dir)
                        window_insert_rootdir.close()
                    elif values['Ladder Test']==True:
                        root_dir = values0[0]
                        Data_Analyser.plot_the_trajectories_ladder(root_dir)
                        window_insert_rootdir.close()
                    break
                elif event == sg.WIN_CLOSED or event == 'Cancel':
                    window_insert_rootdir.close()
                    break
    
        # [Plot the learning curve] pressed
        elif event == 'Plot the learning curve':
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

        # [Stat Analysis] pressed
        elif event == 'Stat analysis':
            # Set the excel analysis file path
            layout2 = [[sg.Text('Enter the excel analysis file path:'), sg.InputText(), sg.FileBrowse()],
                       [sg.Button('Ok'), sg.Button('Cancel')]]
            window_insert_excel_path = sg.Window('Please insert the path of the analysis excel file', layout2)
            while True:
                event, values = window_insert_excel_path.read()
                if event == 'Ok' :
                    excel_path = values[0]
                    print(values)
                    Data_Analyser.stats_effect_weeks(excel_path)
                    window_insert_excel_path.close()
                    break
                elif event == sg.WIN_CLOSED or event == 'Cancel':
                    window_insert_excel_path.close()
                    find = True
                    break

    if find:
        break
    
    
    

        