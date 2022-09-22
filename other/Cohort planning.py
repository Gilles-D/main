# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 17:02:42 2022

@author: gilles.DELBECQ
"""

import datetime
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np





date_of_birth = [("28","07","2022"),("15","08","2022")]

Delai_Injection_j = 0
Delai_Implantation_j = 14
Delai_Rest_j = 7
Behavior_j = 3*3+1
Weaning_j=21

dates=[]
iterator=1

for date in date_of_birth:
    DOB = datetime.date(int(date[2]),int(date[1]),int(date[0]))
    Weaning = DOB + datetime.timedelta(days=21)
    Injection = Weaning + datetime.timedelta(days=Delai_Injection_j)
    Implantation = Injection + datetime.timedelta(days=Delai_Implantation_j)
    Behavior_start = Implantation + datetime.timedelta(days=Delai_Rest_j)
    Behavior_end = Behavior_start + datetime.timedelta(days=Behavior_j)

    
    dates.append((str(iterator),DOB,Weaning,Injection,Implantation,Behavior_start,Behavior_end))
    
    print(f"Cohorte {iterator}")
    print(f"Naissance : {DOB}")
    print(f"Weaning : {Weaning}")
    print(f"Injection : {Injection}")
    print(f"Implantation : {Implantation}")
    print(f'Behavior from {Behavior_start} to {Behavior_end}')
    
    
    iterator=iterator+1
    
df = pd.DataFrame(dates,columns=['Cohort','DOB','Weaning','Injection','Implantation','Behavior_start','Behavior_end'])







"""

# project start date
proj_start = df.DOB.min()

# number of days from project start to task start
df['start'] = (df.DOB-proj_start).dt.days# number of days from project start to end of tasks
df['weaning'] = df.start+Weaning_j
df['inj'] = df.weaning+1
df['rest1'] = df.inj+Delai_Implantation_j
df['imp'] = df.rest1+1
df['rest2'] = df.rest1+Delai_Rest_j
df['behavior'] = df.rest2+1


fig, ax = plt.subplots(1, figsize=(16,6))
ax.barh(df.Cohort, Weaning_j,left=df.start) #Weaning

ax.barh(df.Cohort, 1,left=df.weaning) #injection

ax.barh(df.Cohort, Delai_Implantation_j,left=df.inj) #rest

ax.barh(df.Cohort, 1,left=df.rest1) #implantation

ax.barh(df.Cohort, Delai_Rest_j,left=df.imp) #rest

ax.barh(df.Cohort, Behavior_j,left=df.behavior) #behavior

end_num = df.behavior.max() + Behavior_j

# plt.show()


##### TICKS #####
xticks = np.arange(0, end_num, 3)
xticks_labels = pd.date_range(df.DOB.min(), end=df.Behavior_end.max()).strftime("%m/%d")
xticks_minor = np.arange(0, end_num+3, 1)
ax.set_xticks(xticks)

ax.set_xticks(xticks_minor, minor=True)
ax.set_xticklabels(xticks_labels[::3])

plt.show()

"""

"""
fig = x.timeline(df, x_start="Start", x_end="Finish", y="Task")
fig.update_yaxes(autorange="reversed") # otherwise tasks are listed from the bottom up
fig.show()




# project start date
proj_start = df.Start.min()# number of days from project start to task start
df['start_num'] = (df.Start-proj_start).dt.days# number of days from project start to end of tasks
df['end_num'] = (df.End-proj_start).dt.days# days between start and end of each task
df['days_start_to_end'] = df.end_num - df.start_num



DOB = ("01","09","2022")





DOB = date(int(DOB[2]),int(DOB[1]),int(DOB[0]))
Weaning = DOB + datetime.timedelta(days=21)
Injection = Weaning + datetime.timedelta(days=Delai_Injection_j)
Implantation = Injection + datetime.timedelta(days=Delai_Implantation_j)
Behavior_start = Implantation + datetime.timedelta(days=Delai_Rest_j)
Behavior_end = Behavior_start + datetime.timedelta(days=Behavior_j)

"""

