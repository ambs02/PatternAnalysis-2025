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

# -------------------------------
# Dice Coefficient
# -------------------------------
def dice_coefficient(pred, target, num_classes=6, epsilon=1e-5):
    """Multi-class Dice Similarity Coefficient"""
    pred = torch.argmax(pred, dim=1)  # [B, D, H, W]
    dice_scores = []
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = torch.sum(pred_cls * target_cls)
        union = torch.sum(pred_cls) + torch.sum(target_cls)
        dice = (2 * intersection + epsilon) / (union + epsilon)
        dice_scores.append(dice.item())
    return dice_scores
