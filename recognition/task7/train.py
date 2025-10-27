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
    
def _to_one_hot(pred_logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Logits [B,C,D,H,W] -> one-hot [B,C,D,H,W] via argmax."""
    pred = torch.argmax(pred_logits, dim=1)                      # [B,D,H,W]
    pred_oh = torch.nn.functional.one_hot(pred, num_classes)     # [B,D,H,W,C]
    return pred_oh.permute(0, 4, 1, 2, 3).float()


def _multiclass_dice_from_oh(y_true_oh: torch.Tensor, y_pred_oh: torch.Tensor, eps=1e-5) -> float:
    """
    Multiclass dice coefficient as defined in TF reference:
    mdsc = (2/6) * sum_c ( (|Y∩P| + eps) / (|Y| + |P| + eps) )
    """
    # Reduce over batch and spatial dims
    dims = (0, 2, 3, 4)
    inter = torch.sum(y_true_oh * y_pred_oh, dim=dims)           # [C]
    y_sum = torch.sum(y_true_oh, dim=dims)                       # [C]
    p_sum = torch.sum(y_pred_oh, dim=dims)                       # [C]
    per_class = (inter + eps) / (y_sum + p_sum + eps)            # [C]
    mdsc = (2.0 / y_true_oh.shape[1]) * torch.sum(per_class)     # scalar
    return mdsc.item()


def dice_per_class_from_oh(y_true_oh: torch.Tensor, y_pred_oh: torch.Tensor, eps=1e-5) -> np.ndarray:
    """Return per-class Dice (C,) using 2*|∩|/(|Y|+|P|) over batch+spatial dims."""
    dims = (0, 2, 3, 4)
    inter = torch.sum(y_true_oh * y_pred_oh, dim=dims)           # [C]
    y_sum = torch.sum(y_true_oh, dim=dims)                       # [C]
    p_sum = torch.sum(y_pred_oh, dim=dims)                       # [C]
    dsc = (2.0 * inter + eps) / (y_sum + p_sum + eps)            # [C]
    return dsc.detach().cpu().numpy()

def multiclass_dice_coefficient(y_true_oh: torch.Tensor, y_pred_logits: torch.Tensor) -> float:
    """TF-style function signature name, but in PyTorch."""
    y_pred_oh = _to_one_hot(y_pred_logits, NUM_CLASSES)
    return _multiclass_dice_from_oh(y_true_oh, y_pred_oh)


def dice_coefficient(y_true_oh: torch.Tensor, y_pred_logits: torch.Tensor, class_number: int) -> float:
    """Per-class Dice (matches TF helper style)."""
    y_pred_oh = _to_one_hot(y_pred_logits, NUM_CLASSES)
    dsc = dice_per_class_from_oh(y_true_oh, y_pred_oh)
    return float(dsc[class_number])


def background_dsc(y_true_oh, y_pred_logits): return dice_coefficient(y_true_oh, y_pred_logits, 0)
def body_dsc      (y_true_oh, y_pred_logits): return dice_coefficient(y_true_oh, y_pred_logits, 1)
def bone_dsc      (y_true_oh, y_pred_logits): return dice_coefficient(y_true_oh, y_pred_logits, 2)
def bladder_dsc   (y_true_oh, y_pred_logits): return dice_coefficient(y_true_oh, y_pred_logits, 3)
def rectum_dsc    (y_true_oh, y_pred_logits): return dice_coefficient(y_true_oh, y_pred_logits, 4)
def prostate_dsc  (y_true_oh, y_pred_logits): return dice_coefficient(y_true_oh, y_pred_logits, 5)



def _prepare_targets(lbls: torch.Tensor) -> torch.Tensor:
    # Dataset returns [1, D, H, W] labels; squeeze channel for CE
    if lbls.ndim == 5 and lbls.shape[1] == 1:
        lbls = lbls.squeeze(1)
    return lbls.long()

def run_epoch(model, loader, optimizer, criterion, training: bool):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_voxels = 0

    # For TF-style metric tracking
    mc_dice_vals = []
    per_class_dice_collect = []

    with torch.enable_grad() if training else torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)               # [B,1,D,H,W]
            lbls = _prepare_targets(lbls).to(DEVICE)  # [B,D,H,W]

            logits = model(imgs)                 # [B,C,D,H,W]
            loss = criterion(logits, lbls)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            # Accuracy (per-voxel)
            pred = torch.argmax(logits, dim=1)   # [B,D,H,W]
            total_correct += (pred == lbls).sum().item()
            total_voxels += lbls.numel()

            # One-hot GT for dice metrics
            gt_oh = torch.nn.functional.one_hot(lbls, NUM_CLASSES).permute(0, 4, 1, 2, 3).float()

            # Multiclass dice
            mc_dice_vals.append(multiclass_dice_coefficient(gt_oh, logits))

            # Per-class dice
            per_class_dice_collect.append(dice_per_class_from_oh(gt_oh, _to_one_hot(logits, NUM_CLASSES)))

    mean_loss = total_loss / max(1, len(loader))
    acc = total_correct / max(1, total_voxels)
    mean_mc_dice = float(np.mean(mc_dice_vals)) if mc_dice_vals else 0.0
    mean_per_class_dice = np.mean(np.stack(per_class_dice_collect, axis=0), axis=0) if per_class_dice_collect else np.zeros(NUM_CLASSES)

    return mean_loss, acc, mean_mc_dice, mean_per_class_dice
