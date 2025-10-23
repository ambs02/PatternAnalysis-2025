# images_root = "/Users/amberchen/Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-/data/HipMRI_study_complete_release_v1/semantic_MRs_anon"
# labels_root = "/Users/amberchen/Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-/data/HipMRI_study_complete_release_v1/semantic_labels_anon"

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

from modules import ImprovedUNet3D
from dataset import Prostate3DDataset, discover_pairs

# -------------------------------
# Config
# -------------------------------
DATA_ROOT_IMAGES = "/Users/amberchen/Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-/data/HipMRI_study_complete_release_v1/semantic_MRs_anon"
DATA_ROOT_LABELS = "/Users/amberchen/Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-/data/HipMRI_study_complete_release_v1/semantic_labels_anon"

SAVE_PATH = "./results/"
os.makedirs(SAVE_PATH, exist_ok=True)

EPOCHS = 10
BATCH_SIZE = 2
LR = 1e-4
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
