# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:36:59 2023

@author: Gilles.DELBECQ
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Import the data from the Excel file
data = pd.read_excel("//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/0022_01_08_Mocap_Rates_catwalk_raw_norm_traj.xlsx")


#%% One animated plot
x_data = 'right_foot_x_norm'
y_data = 'right_foot_z_norm'
time_reset = 300  # Number of frames before a point is cleared

# Drop rows where either x_data or y_data is NaN
cleaned_data = data.dropna(subset=[x_data, y_data])

# Create the animation with the cleaned data and clearing old points based on time_reset
fig, ax = plt.subplots(figsize=(10, 6))
plotted_points = []

def init():
    ax.set_xlim(min(cleaned_data[x_data]), max(cleaned_data[x_data]))
    ax.set_ylim(min(cleaned_data[y_data]), max(cleaned_data[y_data]))
    ax.set_xlabel(x_data)
    ax.set_ylabel(y_data)
    ax.set_title(f"Evolution of {y_data} vs {x_data} over time with clearing old points")
    return []

def update(frame):
    global plotted_points
    
    # Plot the new point
    x = cleaned_data[x_data].iloc[frame]
    y = cleaned_data[y_data].iloc[frame]
    point, = ax.plot(x, y, 'b.', alpha=0.5)
    
    # Add the new point to the list of plotted points
    plotted_points.append(point)
    
    # If we have more points than time_reset, remove the oldest point
    if len(plotted_points) > time_reset:
        oldest_point = plotted_points.pop(0)
        oldest_point.remove()
    
    return plotted_points

ani = FuncAnimation(fig, update, frames=len(cleaned_data), init_func=init, blit=True, interval=10)

plt.show()




#%% Two animated plots
# Load and clean the data

x_data = 'right_foot_x_norm'
y_data = 'right_foot_z_norm'
new_x_data = 'right_foot_x'
new_y_data = 'right_foot_z'
time_reset = 250  # Number of frames before a point is cleared

cleaned_data = data.dropna(subset=[x_data, y_data, new_x_data, new_y_data])



# Create a subplot layout with two vertical axes without sharing the x-axis
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
plotted_points_ax1 = []
plotted_points_ax2 = []

def init():
    ax1.set_xlim(min(cleaned_data[x_data]), max(cleaned_data[x_data]))
    ax1.set_ylim(min(cleaned_data[y_data]), max(cleaned_data[y_data]))
    ax1.set_xlabel(x_data)
    ax1.set_ylabel(y_data)
    ax1.set_title(f"Evolution of {y_data} vs {x_data} over time with clearing old points")

    ax2.set_xlim(min(cleaned_data[new_x_data]), max(cleaned_data[new_x_data]))
    ax2.set_ylim(min(cleaned_data[new_y_data]), max(cleaned_data[new_y_data]))
    ax2.set_xlabel(new_x_data)
    ax2.set_ylabel(new_y_data)
    ax2.set_title(f"Evolution of {new_y_data} vs {new_x_data} over time with clearing old points")
    return []

def update(frame):
    global plotted_points_ax1, plotted_points_ax2
    
    # Plotting for the first axis (ax1)
    x1 = cleaned_data[x_data].iloc[frame]
    y1 = cleaned_data[y_data].iloc[frame]
    point1, = ax1.plot(x1, y1, 'b.', alpha=0.5)
    plotted_points_ax1.append(point1)
    if len(plotted_points_ax1) > time_reset:
        oldest_point1 = plotted_points_ax1.pop(0)
        oldest_point1.remove()
    
    # Plotting for the second axis (ax2)
    x2 = cleaned_data[new_x_data].iloc[frame]
    y2 = cleaned_data[new_y_data].iloc[frame]
    point2, = ax2.plot(x2, y2, 'r.', alpha=0.5)
    plotted_points_ax2.append(point2)
    if len(plotted_points_ax2) > time_reset:
        oldest_point2 = plotted_points_ax2.pop(0)
        oldest_point2.remove()

    return []

ani = FuncAnimation(fig, update, frames=len(cleaned_data), init_func=init, blit=True, interval=10)

plt.tight_layout()
plt.show()

ani.save('//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/plots/animation2.gif',writer='pillow', fps=100)


#%%
from matplotlib.widgets import Slider

# Load and clean the data

x_data = 'right_foot_x_norm'
y_data = 'right_foot_z_norm'
new_x_data = 'right_foot_x'
new_y_data = 'right_foot_z'
time_reset = 250  # Number of frames before a point is cleared

cleaned_data = data.dropna(subset=[x_data, y_data, new_x_data, new_y_data])


# Create a subplot layout with two vertical axes without sharing the x-axis and an additional axis for the slider
fig, (ax1, ax2, ax_slider) = plt.subplots(nrows=3, ncols=1, figsize=(10, 14), gridspec_kw={'height_ratios': [1, 1, 0.05]})
plotted_points_ax1 = []
plotted_points_ax2 = []

def init():
    ax1.set_xlim(min(cleaned_data[x_data]), max(cleaned_data[x_data]))
    ax1.set_ylim(min(cleaned_data[y_data]), max(cleaned_data[y_data]))
    ax1.set_xlabel(x_data)
    ax1.set_ylabel(y_data)
    ax1.set_title(f"Evolution of {y_data} vs {x_data} over time with clearing old points")

    ax2.set_xlim(min(cleaned_data[new_x_data]), max(cleaned_data[new_x_data]))
    ax2.set_ylim(min(cleaned_data[new_y_data]), max(cleaned_data[new_y_data]))
    ax2.set_xlabel(new_x_data)
    ax2.set_ylabel(new_y_data)
    ax2.set_title(f"Evolution of {new_y_data} vs {new_x_data} over time with clearing old points")
    return []

def update(val):
    frame = int(slider.val)
    
    global plotted_points_ax1, plotted_points_ax2
    
    # Clear existing points
    for point in plotted_points_ax1:
        point.remove()
    for point in plotted_points_ax2:
        point.remove()
    plotted_points_ax1 = []
    plotted_points_ax2 = []
    
    start_frame = max(0, frame - 50)
    end_frame = min(len(cleaned_data), frame + 51)
    
    for i in range(start_frame, end_frame):
        # Plotting for the first axis (ax1)
        x1 = cleaned_data[x_data].iloc[i]
        y1 = cleaned_data[y_data].iloc[i]
        point1, = ax1.plot(x1, y1, 'b.', alpha=0.5)
        plotted_points_ax1.append(point1)

        # Plotting for the second axis (ax2)
        x2 = cleaned_data[new_x_data].iloc[i]
        y2 = cleaned_data[new_y_data].iloc[i]
        point2, = ax2.plot(x2, y2, 'r.', alpha=0.5)
        plotted_points_ax2.append(point2)
    
    fig.canvas.draw_idle()

# Create the slider
slider = Slider(ax_slider, 'Frame', 0, len(cleaned_data) - 1, valinit=0, valstep=1)
slider.on_changed(update)

init()
plt.tight_layout()
plt.show()






#%%
import numpy as np
right_foot_x = data['right_foot_x']
x_speed = np.abs(np.diff(right_foot_x)/(1/200))

plt.plot(x_speed)
plt.axhline(50)



#%%

import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

# 1. Design the filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

right_foot_x = data['right_foot_x']
right_foot_z = data['right_foot_z']


plt.plot(right_foot_x[0:1000],right_foot_z[0:1000])
plt.plot(butter_lowpass_filter(right_foot_x[0:1000],15,200),butter_lowpass_filter(right_foot_z[0:1000],15,200))
