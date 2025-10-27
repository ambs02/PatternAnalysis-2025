"""
Improved 3D UNet Prediction Script
Generates plasma-colored 3D GIFs for input, true, predicted, and combined visualizations
with fixed aspect ratio and correct framing (no zoom).
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.data import DataLoader

from modules import ImprovedUNet3D
from dataset import Prostate3DDataset, discover_pairs
from train import (
    SAVED_RESULTS_PATH,
    DATA_ROOT_IMAGES,
    DATA_ROOT_LABELS,
    NUM_CLASSES,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = os.path.join(SAVED_RESULTS_PATH, "predictions")
os.makedirs(SAVE_PATH, exist_ok=True)

