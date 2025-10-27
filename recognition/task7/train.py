import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from modules import ImprovedUNet3D
from dataset import Prostate3DDataset, discover_pairs

# Paths / Hyperparameters
DATA_ROOT_IMAGES = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
DATA_ROOT_LABELS = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"
SAVED_RESULTS_PATH = "./results/"
os.makedirs(SAVED_RESULTS_PATH, exist_ok=True)

BATCH_LENGTH = 2
EPOCHS = 10
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 6
CLASS_NAMES = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]

class DiceCELoss(nn.Module):
    def __init__(self, ce_weight=0.5, smooth=1e-5,
                 label_smooth=0.02,
                 ce_class_weights=None,
                 dice_class_weights=None):
        super().__init__()
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.label_smooth = label_smooth
        self.ce = nn.CrossEntropyLoss(weight=ce_class_weights)
        self.dice_class_weights = dice_class_weights

    def forward(self, preds, target):
        ce_loss = self.ce(preds, target)
        probs = torch.softmax(preds, dim=1)
        target_oh = torch.zeros_like(probs).scatter_(1, target.unsqueeze(1), 1)

        if self.label_smooth > 0:
            C = probs.shape[1]
            target_oh = (1 - self.label_smooth) * target_oh + self.label_smooth / C

        dims = (2, 3, 4)
        intersection = torch.sum(probs * target_oh, dim=dims)
        union = torch.sum(probs + target_oh, dim=dims)
        dice_per_c = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss_c = 1.0 - dice_per_c

        if self.dice_class_weights is not None:
            dice_loss = (dice_loss_c * self.dice_class_weights.unsqueeze(0)).mean()
        else:
            dice_loss = dice_loss_c.mean()

        return self.ce_weight * ce_loss + (1.0 - self.ce_weight) * dice_loss