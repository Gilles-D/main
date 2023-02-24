# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 15:25:16 2023

@author: Gilles.DELBECQ
"""

import numpy as np

import matplotlib.pyplot as plt

def phase_correlation(t1, t2):
     """
     Calcule la phase relative entre deux trajectoires en 3D, dans les
trois dimensions.

     Args:
         t1 (ndarray): Un tableau de forme (n_frames, 3) représentant la
première trajectoire.
         t2 (ndarray): Un tableau de forme (n_frames, 3) représentant la
deuxième trajectoire.

     Returns:
         phase (ndarray): Un tableau de forme (3,) contenant les phases
relatives entre les deux trajectoires, dans les trois dimensions.
     """
     import numpy as np
     # Calcule la corrélation croisée entre les deux trajectoires dans chaque dimension
     c = [np.correlate(t1[:, i], t2[:, i], mode='same') for i in range(3)]

     # Détermine l'indice où la corrélation croisée est maximale dans chaque dimension
     peak_idx = [np.argmax(np.abs(c[i])) for i in range(3)]

     # Calcule la phase relative entre les deux trajectoires dans chaque dimension
     
     n_frames = t1.shape[0]
     phase = [(peak_idx[i] - n_frames//2) % n_frames for i in range(3)]

     return np.array(phase)
 
    
 
def phase_correlation_all_frames(window_size, traj1, traj2):
     n_frames = traj1.shape[0]

     phase_all_frames = np.zeros((n_frames - window_size, 3))

     for i in range(n_frames - window_size):
         window1 = traj1[i:i+window_size]
         window2 = traj2[i:i+window_size]

         phase = phase_correlation(window1, window2)

         phase_all_frames[i] = phase

     phase_all_frames = phase_all_frames[window_size//2:window_size//2]

     return phase_all_frames



# # Définir deux trajectoires artificielles sinusoïdales en 3D
# n_frames = 1000
# t = np.linspace(0, 2*np.pi, n_frames)
# traj1 = np.column_stack((np.sin(2*t), np.sin(3*t), np.sin(4*t)))
# traj2 = np.column_stack((np.sin(2*t + np.pi/2), np.sin(3*t + np.pi/3),
# np.sin(4*t + np.pi/4)))

# # Calculer les phases relatives avec une fenêtre de 50 frames
# window_size = 50
# phases = phase_correlation_all_frames(window_size, traj1, traj2)

# # Afficher les résultats
# fig, ax = plt.subplots()
# ax.plot(t[window_size//2:window_size//2], phases[:, 0], label='X')
# ax.plot(t[window_size//2:window_size//2], phases[:, 1], label='Y')
# ax.plot(t[window_size//2:window_size//2], phases[:, 2], label='Z')
# ax.legend()
# ax.set_xlabel('Temps (s)')
# ax.set_ylabel('Phase relative')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_3d_phase(traj1, traj2, window_size):
    n_frames = len(traj1)
    phase_all_frames = np.zeros(n_frames)
    for i in range(window_size//2, n_frames - window_size//2):
        phase_all_frames[i] = np.angle(np.sum(np.exp(1j*np.angle(np.fft.fft(traj1[i-window_size//2:i+window_size//2+1]- traj2[i])))))
    
    phase_all_frames = phase_all_frames[window_size//2:window_size//2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj1[:, 0], traj1[:, 1], traj1[:, 2], label='Trajectory 1')
    ax.plot(traj2[:, 0], traj2[:, 1], traj2[:, 2], label='Trajectory 2')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim(-100, 100) # set the limit of z-axis
    plt.legend()

    fig2 = plt.figure()
    plt.plot(phase_all_frames)
    plt.xlabel('Frame')
    plt.ylabel('Phase')
    plt.show()

# Example
n_frames = 500
t = np.linspace(0, 2*np.pi, n_frames)
traj1 = np.stack([np.sin(t), np.cos(t), np.zeros(n_frames)]).T
traj2 = np.stack([np.cos(t), np.zeros(n_frames), np.sin(t)]).T

plot_3d_phase(traj1, traj2, window_size=50)