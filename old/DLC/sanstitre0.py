# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 18:03:07 2019

@author: Gilles.DELBECQ
"""


from subprocess  import call 

command = "MP4Box -add C:/Users/Gilles.DELBECQ/Desktop/my_beam-gilles-2019-12-16/videos/videotest1.h264 C:/Users/Gilles.DELBECQ/Desktop/my_beam-gilles-2019-12-16/videos/videotest1.mp4"
call([command], shell=True)
print("vid conv")