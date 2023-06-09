# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 16:22:26 2023

@author: MOCAP
"""

import numpy as np
import matplotlib.pyplot as plt


from quantities import ms, s, Hz
from elephant.spike_train_generation import homogeneous_poisson_process, homogeneous_gamma_process

from elephant.statistics import mean_firing_rate

from neo.core import SpikeTrain



spike_times = np.load('C:/Users/MOCAP/Desktop/temp/0012_24_05_all_01_06/phy_export/tridesclous/spike_times.npy')
spike_cluster = np.load('C:/Users/MOCAP/Desktop/temp/0012_24_05_all_01_06/phy_export/tridesclous/spike_clusters.npy')
spike_templates = np.load('C:/Users/MOCAP/Desktop/temp/0012_24_05_all_01_06/phy_export/tridesclous/similar_templates.npy')



#%% Section 1

t_start = 275.5 * ms
print(t_start)

t_start2 = 3. * s
t_start_sum = t_start + t_start2
print(t_start_sum)

np.random.seed(28)  # to make the results reproducible
spiketrain1 = homogeneous_poisson_process(rate=10*Hz, t_start=0.*ms, t_stop=10000.*ms)
spiketrain2 = homogeneous_gamma_process(a=3, b=10*Hz, t_start=0.*ms, t_stop=10000.*ms)




print("spiketrain1 type is", type(spiketrain1))
print("spiketrain2 type is", type(spiketrain2))

print(f"spiketrain2 has {len(spiketrain2)} spikes:")
print("  t_start:", spiketrain2.t_start)
print("  t_stop:", spiketrain2.t_stop)
print("  spike times:", spiketrain2.times)

plt.figure(figsize=(8, 3))
plt.eventplot([spiketrain1.magnitude, spiketrain2.magnitude], linelengths=0.75, color='black')
plt.xlabel('Time (ms)', fontsize=16)
plt.yticks([0,1], labels=["spiketrain1", "spiketrain2"], fontsize=16)
plt.title("Figure 1");


#%% Section 1

print("The mean firing rate of spiketrain1 is", mean_firing_rate(spiketrain1))
print("The mean firing rate of spiketrain2 is", mean_firing_rate(spiketrain2))

fr1 = len(spiketrain1) / (spiketrain1.t_stop - spiketrain1.t_start)
fr2 = len(spiketrain2) / (spiketrain2.t_stop - spiketrain2.t_start)
print("The mean firing rate of spiketrain1 is", fr1)
print("The mean firing rate of spiketrain2 is", fr2)

mean_firing_rate(spiketrain1, t_start=0*ms, t_stop=1000*ms)