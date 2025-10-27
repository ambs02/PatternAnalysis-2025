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


# Metrics
def _to_one_hot_labels(lbls_long: torch.Tensor, num_classes: int) -> torch.Tensor:
    oh = torch.nn.functional.one_hot(lbls_long, num_classes=num_classes)
    return oh.permute(0, 4, 1, 2, 3).float()


@torch.no_grad()
def dice_per_class(pred_logits: torch.Tensor, gt_long: torch.Tensor, num_classes: int = NUM_CLASSES, eps: float = 1e-5):
    probs = torch.softmax(pred_logits, dim=1)
    pred_oh = torch.zeros_like(probs).scatter_(1, probs.argmax(dim=1, keepdim=True), 1)
    gt_oh = _to_one_hot_labels(gt_long, num_classes)
    dims = (0, 2, 3, 4)
    inter = torch.sum(pred_oh * gt_oh, dim=dims)
    denom = torch.sum(pred_oh + gt_oh, dim=dims)
    dice_c = (2.0 * inter + eps) / (denom + eps)
    return dice_c.detach().cpu().numpy()


