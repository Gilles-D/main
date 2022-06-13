# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import deeplabcut
import os

os.chdir(r"C:/Users/Gilles.DELBECQ/Desktop/")
config_path = r'C:/Users/Gilles.DELBECQ/Desktop/colab-gilles-2019-12-19/config.yaml'
videofile_path = os.path.join(os.getcwd(),'C:/Users/Gilles.DELBECQ/Desktop/my_beam-gilles-2019-12-16/videos')


# deeplabcut.create_new_project('colab', 'gilles', [r'C:/Users/Gilles.DELBECQ/Desktop/Videos/mp4/videotest2.mp4'], working_directory=r"C:/Users/Gilles.DELBECQ/Desktop/",copy_videos=True,videotype='.mp4')


deeplabcut.extract_frames(config_path)

deeplabcut.label_frames(config_path)

deeplabcut.check_labels(config_path)

deeplabcut.create_training_dataset(config_path)

deeplabcut.train_network(config_path,shuffle=1, displayiters=5, saveiters=100, maxiters=700)

deeplabcut.evaluate_network(config_path,plotting=False)

deeplabcut.analyze_videos(config_path,[videofile_path],videotype='.mp4')

# deeplabcut.filterpredictions(config_path, [videofile_path],videotype='.mp4')
# deeplabcut.plot_trajectories(config_path, [videofile_path],videotype='.mp4')

deeplabcut.create_labeled_video(config_path, [videofile_path])

deeplabcut.extract_outlier_frames(config_path, [videofile_path],videotype='.mp4',)

deeplabcut.refine_labels(config_path)

deeplabcut.check_labels(config_path)

deeplabcut.create_training_dataset(config_path)

deeplabcut.merge_datasets(config_path)
