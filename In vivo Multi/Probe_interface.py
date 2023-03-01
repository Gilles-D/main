# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:38:44 2023

@author: Gilles.DELBECQ
"""

import numpy as np
import matplotlib.pyplot as plt

from probeinterface import Probe, ProbeGroup,write_probeinterface
from probeinterface.plotting import plot_probe

n = 16


"""
Buzsaki16
"""

positions = ([[-18.5,140.0],
              [-14.5,100.0],
              [-10.5,60.0],
              [-8.5,20.0],
              [0.0,0.0],
              [8.5,40.0],
              [12.5,80.0],
              [16.5,120.0],
              [181.5,140.0],
              [185.5,100.0],
              [189.5,60.0],
              [191.5,20.0],
              [200.0,0.0],
              [208.5,40.0],
              [212.5,80.0],
              [216.5,120.0]])

probe = Probe(ndim=2, si_units='um')
probe.set_contacts(positions=positions, shapes='square', shape_params={'width': 12.65})

print(probe)

polygon = [(0, -22), (-14.5, 15),(-26.5, 140),(-26.5, 200),(225.5, 200),(225.5, 140),(213.5, 15),(200, -22),
           (213.5-29, 15),(225.5-57, 140),(225.5-57, 170),(-26.5+57, 170),(-26.5+57, 140), (-14.5+27, 15)]
probe.set_planar_contour(polygon)

df = probe.to_dataframe()
df

plot_probe(probe,with_channel_index=True)


"""
https://intantech.com/image-src/RHD2132_16ch_electrode_connector_600.jpg
"""

#channel_indices = [11,10,9,8,12,13,14,15,16,17,18,19,23,22,21,20]
channel_indices = [4,3,2,1,5,6,7,8,9,10,11,12,16,15,14,13]

probe.set_device_channel_indices(channel_indices)
print(probe.device_channel_indices)

plot_probe(probe, with_channel_index=True, with_device_index=True)

probegroup = ProbeGroup()
probegroup.add_probe(probe)

write_probeinterface(r'C:/Users/Gilles.DELBECQ/Desktop/Buzsaki16.json', probegroup)


"""
A1x16-Poly2-5mm-50s-177
"""
"""

positions = ([[0, 250],
       [0, 300],
       [0, 350],
       [0, 200],
       [0, 150],
       [0, 100],
       [0, 50],
       [0, 0],
       [43.3, 25],
       [43.3, 75],
       [43.3, 125],
       [43.3, 175],
       [43.3, 225],
       [43.3, 375],
       [43.3, 325],
       [43.3, 275]])

probe = Probe(ndim=2, si_units='um')
probe.set_contacts(positions=positions, shapes='circle', shape_params={'radius': 7.5})

print(probe)

polygon = [(21.65, -75), (-12.35, 0),(-12.35, 450), (55.65, 450), (55.65, 0) ]
probe.set_planar_contour(polygon)

df = probe.to_dataframe()
df

plot_probe(probe,with_channel_index=True)



"""
'https://intantech.com/image-src/RHD2132_16ch_electrode_connector_600.jpg'
"""

channel_indices = [11,10,9,8,12,13,14,15,16,17,18,19,23,22,21,20]
probe.set_device_channel_indices(channel_indices)
print(probe.device_channel_indices)

plot_probe(probe, with_channel_index=True, with_device_index=True)

probegroup = ProbeGroup()
probegroup.add_probe(probe)

write_probeinterface(r'C:/Users/Gilles.DELBECQ/Desktop/A1x16-Poly2-5mm-50s-177.json', probegroup)

"""
