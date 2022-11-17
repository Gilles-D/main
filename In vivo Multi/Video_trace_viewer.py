from matplotvideo import attach_video_player_to_figure
import matplotlib.pyplot as plt
import numpy as np

from scipy import signal

# (timestamp, value) pairs

# sample: big bunny scene cuts
# fancy_data = [
#     (0, 1),
#     (11.875, 1),
#     (11.917, 2),
#     (15.75, 2),
#     (15.792, 3),
#     (23.042, 3),
#     (23.083, 4),
#     (47.708, 4),
#     (47.75, 5),
#     (56.083, 5),
#     (56.125, 6),
#     (60, 6)
# ]

"""
path_cmr=r"//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/Files/RBF/10-24/preprocessed/0004_07_0007_20000Hz_cmr.rbf"
data_cmr=np.fromfile(path_cmr)
data_cmr=data_cmr.reshape(int(len(data_cmr)/6),-1).transpose()

sampling_rate=20000

data_cmr_chan3=data_cmr[3]
time_vector = np.arange(0,len(data_cmr_chan3)/sampling_rate,1/sampling_rate)
array=np.stack((time_vector,data_cmr_chan3),axis=0).transpose()

fancy_data =tuple([tuple(row) for row in array])
"""

path_cmr=r"//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/Files/RBF/10-24/preprocessed/0004_07_0008_20000Hz_cmr.rbf"
data_cmr=np.fromfile(path_cmr)*1000
data_cmr=data_cmr.reshape(int(len(data_cmr)/6),-1).transpose()

sampling_rate=20000
cam_freq=30
shift_cam_frames = 8 #in frames
data_cmr_chan3=data_cmr[3]
downsampled =  signal.resample(data_cmr_chan3, int(len(data_cmr_chan3)/2))
sampling_rate_down=sampling_rate/2

shift_cam = np.zeros(int(shift_cam_frames/cam_freq*sampling_rate_down))
downsampled=np.concatenate((shift_cam,downsampled))

time_vector = np.arange(0,len(downsampled)/sampling_rate_down,1/sampling_rate_down)

array=np.stack((time_vector,downsampled),axis=0).transpose()

fancy_data =tuple([tuple(row) for row in array])


def on_frame(video_timestamp, line):
    timestamps, y = zip(*fancy_data)
    x = [timestamp - video_timestamp for timestamp in timestamps]

    line.set_data(x, y)
    line.axes.relim()
    line.axes.autoscale_view()
    line.axes.figure.canvas.draw()


def main():
    fig, ax = plt.subplots()
    plt.xlim(-1.5, 1.5)
    plt.axvline(x=0, color='k', linestyle='--')

    line, = ax.plot([], [], color='blue')

    attach_video_player_to_figure(fig, "//equipe2-nas1/Gilles.DELBECQ/Data/ePhy/Cohorte 1/Files/video/10-24/04_07_0008.avi", on_frame, line=line)

    plt.show()


main()