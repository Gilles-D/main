# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:51:03 2019

@author: gille
"""


import matplotlib.pyplot as plt
import numpy

labels = ['Neurones répondant à la stimulation', 'Neurones ne répondant pas à la stimulation']
sizes = [11, 27]
colors = ['lightskyblue', 'sandybrown']


def absolute_value(val):
    a  = numpy.round(val/100.*numpy.array(sizes).sum())
    return a

explode = (0.03, 0)  # explode 1st slice

patches, texts = plt.pie(sizes,explode=explode, colors=colors, shadow=False, startangle=90)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()