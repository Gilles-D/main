# -*- coding: utf-8 -*-


video='Cam_Right June 1228 J12 6mm R 1.avi'


import os

rootdir = r'D:/Gilles.DELBECQ/Manips/Beam Test/Mise en place Juin 2020/Raw Videos'


for subdirs, dirs, files in os.walk(rootdir):
    for file in files:
        file_path = os.path.join(subdirs, file) 
        file_split = file.split(' ')
        
        if file_split[3]=='J1' and file_split[4]=='12mm' and file_split[5]=='C':
            Session = 1
        if file_split[3]=='J1' and file_split[4]=='10mm' and file_split[5]=='C':
            Session = 2
        if file_split[3]=='J2' and file_split[4]=='10mm' and file_split[5]=='C':
            Session = 3
        if file_split[3]=='J2' and file_split[4]=='10mm' and file_split[5]=='R':
            Session = 4
        if file_split[3]=='J3' and file_split[4]=='10mm' and file_split[5]=='R':
            Session = 5
        if file_split[3]=='J4' and file_split[4]=='10mm' and file_split[5]=='R':
            Session = 6
        if file_split[3]=='J4' and file_split[4]=='8mm' and file_split[5]=='R':
            Session = 7
        if file_split[3]=='J5' and file_split[4]=='8mm' and file_split[5]=='R':
            Session = 8
        if file_split[3]=='J5' and file_split[4]=='6mm' and file_split[5]=='R':
            Session = 9
        if file_split[3]=='J6' and file_split[4]=='6mm' and file_split[5]=='R':
            Session = 10
        if file_split[3]=='J7' and file_split[4]=='6mm' and file_split[5]=='R':
            Session = 11
        if file_split[3]=='J8' and file_split[4]=='6mm' and file_split[5]=='R':
            Session = 12
        if file_split[3]=='J9' and file_split[4]=='6mm' and file_split[5]=='R':
            Session = 13
        if file_split[3]=='J10' and file_split[4]=='6mm' and file_split[5]=='R':
            Session = 14
        if file_split[3]=='J11' and file_split[4]=='6mm' and file_split[5]=='R':
            Session = 15
        if file_split[3]=='J12' and file_split[4]=='6mm' and file_split[5]=='R':
            Session = 16
        if file_split[3]=='J13' and file_split[4]=='6mm' and file_split[5]=='R':
            Session = 17
        if file_split[3]=='J14' and file_split[4]=='6mm' and file_split[5]=='R':
            Session = 18
        if file_split[3]=='J15' and file_split[4]=='6mm' and file_split[5]=='R':
            Session = 19
        if file_split[3]=='J16' and file_split[4]=='6mm' and file_split[5]=='R':
            Session = 20
        if file_split[3]=='J17' and file_split[4]=='6mm' and file_split[5]=='R':
            Session = 21                
        if file_split[3]=='J18' and file_split[4]=='6mm' and file_split[5]=='R':
            Session = 22 
        if file_split[3]=='J19' and file_split[4]=='6mm' and file_split[5]=='R':
            Session = 23
        if file_split[3]=='J20' and file_split[4]=='6mm' and file_split[5]=='R':
            Session = 24
        
        trial = file_split[6].split('.')[0]
        
        new_file = '{} {} Session {} {}.avi'.format(file_split[1], file_split[2], Session, trial)
        new_path = os.path.join(subdirs, new_file) 
        os.rename(file_path,new_path)