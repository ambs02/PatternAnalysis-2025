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


# Visualisation helpers
def _normalize_image(volume):
    """Normalize MRI to [0,1] with stable contrast."""
    vol = np.array(volume, dtype=np.float32)
    mean = np.mean(vol)
    std = np.std(vol) + 1e-8
    vol = (vol - mean) / std
    vol = np.clip(vol, -2.0, 3.0)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    return vol


def _prepare_volume_for_show(x: np.ndarray):
    vol = np.squeeze(np.array(x))
    if vol.ndim == 4 and vol.shape[0] == 1:
        vol = vol[0]
    if vol.ndim == 3 and vol.shape[0] < vol.shape[-1]:
        vol = np.moveaxis(vol, -1, 0)
    return vol


def save_volume_gif(volume_3d, title, index):
    """Save a single 3D GIF with plasma color and full axis display."""
    os.makedirs(SAVE_PATH, exist_ok=True)
    volume_3d = _normalize_image(_prepare_volume_for_show(volume_3d))
    cmap, vmin, vmax = "plasma", 0.05, 0.95

    fig, ax = plt.subplots(figsize=(4, 6))  # tall, not zoomed in
    ims = []
    step = max(1, volume_3d.shape[0] // 64)
    slices = range(0, volume_3d.shape[0], step)

    for i in slices:
        frame = volume_3d[i, :, :]
        im = ax.imshow(
            frame,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            animated=True,
            aspect="auto",          # fills axis area (not distorted)
            interpolation="nearest",
            origin="upper"
        )
        ims.append([im])

    # keep ticks and axis labels
    ax.set_title(title, fontsize=14, pad=5)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # use tight layout to ensure full use of figure space
    plt.tight_layout(pad=0.5)

    ani = animation.ArtistAnimation(fig, ims, interval=60, blit=True, repeat_delay=1000)
    out_path = os.path.join(SAVE_PATH, f"{title}_{index}.gif")
    try:
        ani.save(out_path, writer="ffmpeg", fps=15, dpi=150)
    except Exception:
        ani.save(out_path, writer="pillow", fps=15)
    plt.close(fig)
    print(f"Saved {title} animation to {out_path}")


def save_combined_gif(img, true_mask, pred_mask, index):
    """Side-by-side comparison (Input | True | Predicted) with plasma color and full axis."""
    os.makedirs(SAVE_PATH, exist_ok=True)
    img = _normalize_image(img)
    true_mask = _normalize_image(true_mask)
    pred_mask = _normalize_image(pred_mask)

    img = _prepare_volume_for_show(img)
    true_mask = _prepare_volume_for_show(true_mask)
    pred_mask = _prepare_volume_for_show(pred_mask)

    step = max(1, img.shape[0] // 64)
    slices = range(0, img.shape[0], step)
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    titles = ["Input", "True", "Predicted"]
    ims = []

    for a, t in zip(axes, titles):
        a.set_title(t, fontsize=14)
        a.set_xlabel("Width")
        a.set_ylabel("Height")

    for i in slices:
        frame1 = axes[0].imshow(img[i], cmap="plasma", vmin=0.05, vmax=0.95,
                                animated=True, aspect="auto", interpolation="nearest", origin="upper")
        frame2 = axes[1].imshow(true_mask[i], cmap="plasma", vmin=0.05, vmax=0.95,
                                animated=True, aspect="auto", interpolation="nearest", origin="upper")
        frame3 = axes[2].imshow(pred_mask[i], cmap="plasma", vmin=0.05, vmax=0.95,
                                animated=True, aspect="auto", interpolation="nearest", origin="upper")
        ims.append([frame1, frame2, frame3])

    plt.tight_layout(pad=1.0)

    ani = animation.ArtistAnimation(fig, ims, interval=60, blit=True, repeat_delay=1000)
    out_path = os.path.join(SAVE_PATH, f"Combined_{index}.gif")
    try:
        ani.save(out_path, writer="ffmpeg", fps=15, dpi=150)
    except Exception:
        ani.save(out_path, writer="pillow", fps=15)
    plt.close(fig)
    print(f"Saved combined animation to {out_path}")


# Core prediction
@torch.no_grad()
def display_and_save_examples(dataset, number_of_examples, model, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    for count, (imgs, lbls) in enumerate(loader):
        if lbls.ndim == 5 and lbls.shape[1] == 1:
            lbls = lbls.squeeze(1)
        imgs, lbls = imgs.to(device), lbls.long().to(device)

        logits = model(imgs)
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)

        dpc = dice_per_class(logits, lbls)
        mdc = float(dpc.mean())
        print(f"\n Example {count+1}:")
        print(f"  Multiclass Dice: {mdc:.4f}")
        class_names = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]
        for i, name in enumerate(class_names):
            print(f"  {name} DSC: {dpc[i]:.4f}")

        img_np = imgs.detach().cpu().numpy()[0, 0]
        true_np = lbls.detach().cpu().numpy()[0]
        pred_np = pred.detach().cpu().numpy()[0]

        save_volume_gif(img_np, "InputImage", count)
        save_volume_gif(true_np, "TrueMask", count)
        save_volume_gif(pred_np, "PredictedMask", count)
        save_combined_gif(img_np, true_np, pred_np, count)

        if count + 1 >= number_of_examples:
            break


