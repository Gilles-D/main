# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import deeplabcut
import os

os.chdir(r"D:/beam_test")


deeplabcut.create_new_project('beam', 'gilles', [r'D:/beam_test/video_wt.mp4'], working_directory=r"D:/beam_test",copy_videos=True,videotype='.mp4')

config_path = r'D:/beam_test/beam-gilles-2019-12-14/config.yaml'
videofile_path = os.path.join(os.getcwd(),'beam-gilles-2019-12-14/videos/video_wt.mp4')

# deeplabcut.extract_frames(config_path)

# deeplabcut.label_frames(config_path)

# deeplabcut.check_labels(config_path)

deeplabcut.create_training_dataset(config_path)

deeplabcut.train_network(config_path,shuffle=1, displayiters=1, saveiters=100, gputouse = 0,allow_growth = True)

deeplabcut.evaluate_network(config_path,plotting=False)

deeplabcut.analyze_videos(config_path,[videofile_path],videotype='.mp4',gputouse=0)

# deeplabcut.filterpredictions(config_path, [videofile_path],videotype='.mp4')
# deeplabcut.plot_trajectories(config_path, [videofile_path],videotype='.mp4')

deeplabcut.create_labeled_video(config_path, [videofile_path])

deeplabcut.extract_outlier_frames(config_path, [videofile_path],videotype='.mp4',)

deeplabcut.refine_labels(config_path)

deeplabcut.check_labels(config_path)

deeplabcut.merge_datasets(config_path)
