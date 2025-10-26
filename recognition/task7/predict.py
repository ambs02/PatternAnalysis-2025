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


# -------------------------------
# Metrics
# -------------------------------
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


# -------------------------------
# Visualization helpers
# -------------------------------
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


def _make_axis_clean(ax):
    """Remove borders, labels, and fill figure."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])  # Fill subplot fully


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
    print(f"✅ Saved {title} animation to {out_path}")


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
    print(f"✅ Saved combined animation to {out_path}")



# -------------------------------
# Core prediction
# -------------------------------
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
        print(f"\n🧩 Example {count+1}:")
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


def predict_model():
    print("🚀 Loading trained model...")
    model_path = os.path.join(SAVED_RESULTS_PATH, "improved_3d_unet_model.pth")
    model = ImprovedUNet3D().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"✅ Loaded model from {model_path}")

    imgs, lbls = discover_pairs(DATA_ROOT_IMAGES, DATA_ROOT_LABELS)
    print(f"✅ Found {len(imgs)} matching pairs between '{DATA_ROOT_IMAGES}' and '{DATA_ROOT_LABELS}'.")
    test_dataset = Prostate3DDataset(imgs, lbls, downsample=(0.5, 0.5, 0.5), augment=False)
    print(f"🧠 Dataset size: {len(test_dataset)} samples")

    display_and_save_examples(test_dataset, 3, model, DEVICE)


if __name__ == "__main__":
    predict_model()




# """
# PyTorch prediction script for Improved 3D UNet model.
# Generates predictions and saves 3D animated GIFs for input, true, predicted,
# and combined (side-by-side) visualizations.

# Author: Nathan King (PyTorch port, enhanced visualization, plasma color version)
# """

# import os
# import numpy as np
# import torch
# import torch.nn.functional as F
# import matplotlib
# matplotlib.use("Agg")  # safe for remote/HPC
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from torch.utils.data import DataLoader

# from modules import ImprovedUNet3D
# from dataset import Prostate3DDataset, discover_pairs

# # -------------------------------
# # Import config from train.py
# # -------------------------------
# from train import (
#     SAVED_RESULTS_PATH,
#     DATA_ROOT_IMAGES,
#     DATA_ROOT_LABELS,
#     NUM_CLASSES,
# )

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SAVE_PATH = os.path.join(SAVED_RESULTS_PATH, "predictions")
# os.makedirs(SAVE_PATH, exist_ok=True)

# # -------------------------------
# # Metrics
# # -------------------------------
# def _to_one_hot_labels(lbls_long: torch.Tensor, num_classes: int) -> torch.Tensor:
#     oh = torch.nn.functional.one_hot(lbls_long, num_classes=num_classes)  # [B,D,H,W,C]
#     return oh.permute(0, 4, 1, 2, 3).float()  # [B,C,D,H,W]

# @torch.no_grad()
# def dice_per_class(pred_logits: torch.Tensor, gt_long: torch.Tensor, num_classes: int = NUM_CLASSES, eps: float = 1e-5):
#     probs = torch.softmax(pred_logits, dim=1)
#     pred_oh = torch.zeros_like(probs).scatter_(1, probs.argmax(dim=1, keepdim=True), 1)
#     gt_oh = _to_one_hot_labels(gt_long, num_classes)
#     dims = (0, 2, 3, 4)
#     inter = torch.sum(pred_oh * gt_oh, dim=dims)
#     denom = torch.sum(pred_oh + gt_oh, dim=dims)
#     dice_c = (2.0 * inter + eps) / (denom + eps)
#     return dice_c.detach().cpu().numpy()

# # -------------------------------
# # Visualization helpers (plasma version)
# # -------------------------------
# def _normalize_image(volume):
#     """
#     Normalize a 3D volume to [0,1] for visualization.
#     Keeps consistent contrast and prevents extreme outliers.
#     """
#     vol = np.array(volume, dtype=np.float32)
#     mean = np.mean(vol)
#     std = np.std(vol) + 1e-8
#     vol = (vol - mean) / std
#     vol = np.clip(vol, -2.0, 3.0)
#     vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
#     return vol

# def _prepare_volume_for_show(x: np.ndarray):
#     vol = np.squeeze(np.array(x))
#     if vol.ndim == 4 and vol.shape[0] == 1:
#         vol = vol[0]
#     if vol.ndim == 3 and vol.shape[0] < vol.shape[-1]:
#         vol = np.moveaxis(vol, -1, 0)
#     return vol

# def save_volume_gif(volume_3d, title, index):
#     """Save a plasma-colored GIF for any 3D volume (input, true, or predicted)."""
#     os.makedirs(SAVE_PATH, exist_ok=True)
#     volume_3d = _normalize_image(_prepare_volume_for_show(volume_3d))

#     cmap, vmin, vmax = "plasma", 0.05, 0.95

#     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#     ims = []
#     step = max(1, volume_3d.shape[0] // 64)
#     slices = range(0, volume_3d.shape[0], step)

#     for i in slices:
#         frame = volume_3d[i, :, :]
#         im = ax.imshow(frame, cmap=cmap, vmin=vmin, vmax=vmax,
#                        animated=True, aspect='auto', interpolation='nearest', origin='upper')
#         if i == slices.start:
#             ax.set_title(title, fontsize=14)
#             ax.axis("off")
#             ax.set_position([0, 0, 1, 1])  # fill figure
#         ims.append([im])

#     ani = animation.ArtistAnimation(fig, ims, interval=60, blit=True, repeat_delay=1000)
#     out_path = os.path.join(SAVE_PATH, f"{title}_{index}.gif")
#     try:
#         ani.save(out_path, writer="ffmpeg", fps=15, dpi=150)
#     except Exception:
#         ani.save(out_path, writer="pillow", fps=15)
#     plt.close(fig)
#     print(f"✅ Saved {title} animation to {out_path}")

# def save_combined_gif(img, true_mask, pred_mask, index):
#     """Side-by-side comparison (Input | True | Predicted), all in plasma colormap."""
#     os.makedirs(SAVE_PATH, exist_ok=True)
#     img = _normalize_image(img)
#     true_mask = _normalize_image(true_mask)
#     pred_mask = _normalize_image(pred_mask)

#     img = _prepare_volume_for_show(img)
#     true_mask = _prepare_volume_for_show(true_mask)
#     pred_mask = _prepare_volume_for_show(pred_mask)

#     step = max(1, img.shape[0] // 64)
#     slices = range(0, img.shape[0], step)
#     fig, axes = plt.subplots(1, 3, figsize=(12, 5))
#     titles = ["Input", "True", "Predicted"]
#     ims = []

#     for a, t in zip(axes, titles):
#         a.set_title(t, fontsize=14)
#         a.axis("off")
#         a.set_position([0, 0, 1, 1])  # fill subplot cell

#     for i in slices:
#         frame1 = axes[0].imshow(img[i], cmap="plasma", vmin=0.05, vmax=0.95,
#                                 animated=True, aspect='auto', interpolation='nearest', origin='upper')
#         frame2 = axes[1].imshow(true_mask[i], cmap="plasma", vmin=0.05, vmax=0.95,
#                                 animated=True, aspect='auto', interpolation='nearest', origin='upper')
#         frame3 = axes[2].imshow(pred_mask[i], cmap="plasma", vmin=0.05, vmax=0.95,
#                                 animated=True, aspect='auto', interpolation='nearest', origin='upper')
#         ims.append([frame1, frame2, frame3])

#     ani = animation.ArtistAnimation(fig, ims, interval=60, blit=True, repeat_delay=1000)
#     out_path = os.path.join(SAVE_PATH, f"Combined_{index}.gif")
#     try:
#         ani.save(out_path, writer="ffmpeg", fps=15, dpi=150)
#     except Exception:
#         ani.save(out_path, writer="pillow", fps=15)
#     plt.close(fig)
#     print(f"✅ Saved combined animation to {out_path}")

# # -------------------------------
# # Core prediction
# # -------------------------------
# @torch.no_grad()
# def display_and_save_examples(dataset, number_of_examples, model, device):
#     model.eval()
#     loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

#     for count, (imgs, lbls) in enumerate(loader):
#         if lbls.ndim == 5 and lbls.shape[1] == 1:
#             lbls = lbls.squeeze(1)
#         imgs, lbls = imgs.to(device), lbls.long().to(device)

#         logits = model(imgs)
#         probs = F.softmax(logits, dim=1)
#         pred = probs.argmax(dim=1)

#         dpc = dice_per_class(logits, lbls)
#         mdc = float(dpc.mean())
#         print(f"\n🧩 Example {count+1}:")
#         print(f"  Multiclass Dice: {mdc:.4f}")
#         class_names = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]
#         for i, name in enumerate(class_names):
#             print(f"  {name} DSC: {dpc[i]:.4f}")

#         img_np = imgs.detach().cpu().numpy()[0, 0]
#         true_np = lbls.detach().cpu().numpy()[0]
#         pred_np = pred.detach().cpu().numpy()[0]

#         save_volume_gif(img_np, "InputImage", count)
#         save_volume_gif(true_np, "TrueMask", count)
#         save_volume_gif(pred_np, "PredictedMask", count)
#         save_combined_gif(img_np, true_np, pred_np, count)

#         if count + 1 >= number_of_examples:
#             break

# def predict_model():
#     print("🚀 Loading trained model...")
#     model_path = os.path.join(SAVED_RESULTS_PATH, "improved_3d_unet_model.pth")
#     model = ImprovedUNet3D().to(DEVICE)
#     model.load_state_dict(torch.load(model_path, map_location=DEVICE))
#     model.eval()
#     print(f"✅ Loaded model from {model_path}")

#     imgs, lbls = discover_pairs(DATA_ROOT_IMAGES, DATA_ROOT_LABELS)
#     print(f"✅ Found {len(imgs)} matching pairs between '{DATA_ROOT_IMAGES}' and '{DATA_ROOT_LABELS}'.")
#     test_dataset = Prostate3DDataset(imgs, lbls, downsample=(0.5, 0.5, 0.5), augment=False)
#     print(f"🧠 Dataset size: {len(test_dataset)} samples")

#     display_and_save_examples(test_dataset, 3, model, DEVICE)

# if __name__ == "__main__":
#     predict_model()


# """
# PyTorch prediction script for Improved 3D UNet model.
# Generates predictions and saves 3D animated GIFs for input, true, predicted,
# and combined (side-by-side) visualizations.

# Author: Nathan King (PyTorch port, enhanced visualization, color version)
# """

# import os
# import numpy as np
# import torch
# import torch.nn.functional as F
# import matplotlib
# matplotlib.use("Agg")  # safe for remote/HPC
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from torch.utils.data import DataLoader

# from modules import ImprovedUNet3D
# from dataset import Prostate3DDataset, discover_pairs

# # -------------------------------
# # Import config from train.py
# # -------------------------------
# from train import (
#     SAVED_RESULTS_PATH,
#     DATA_ROOT_IMAGES,
#     DATA_ROOT_LABELS,
#     NUM_CLASSES,
# )

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SAVE_PATH = os.path.join(SAVED_RESULTS_PATH, "predictions")
# os.makedirs(SAVE_PATH, exist_ok=True)

# # -------------------------------
# # Metrics
# # -------------------------------
# def _to_one_hot_labels(lbls_long: torch.Tensor, num_classes: int) -> torch.Tensor:
#     oh = torch.nn.functional.one_hot(lbls_long, num_classes=num_classes)  # [B,D,H,W,C]
#     return oh.permute(0, 4, 1, 2, 3).float()  # [B,C,D,H,W]

# @torch.no_grad()
# def dice_per_class(pred_logits: torch.Tensor, gt_long: torch.Tensor, num_classes: int = NUM_CLASSES, eps: float = 1e-5):
#     probs = torch.softmax(pred_logits, dim=1)
#     pred_oh = torch.zeros_like(probs).scatter_(1, probs.argmax(dim=1, keepdim=True), 1)
#     gt_oh = _to_one_hot_labels(gt_long, num_classes)
#     dims = (0, 2, 3, 4)
#     inter = torch.sum(pred_oh * gt_oh, dim=dims)
#     denom = torch.sum(pred_oh + gt_oh, dim=dims)
#     dice_c = (2.0 * inter + eps) / (denom + eps)
#     return dice_c.detach().cpu().numpy()

# # -------------------------------
# # Visualization helpers (color MRI version)
# # -------------------------------
# def _normalize_image(volume):
#     """
#     Intensity-normalize a 3D MRI volume for visualization with consistent global contrast.
#     """
#     vol = np.array(volume, dtype=np.float32)
#     mean = np.mean(vol)
#     std = np.std(vol) + 1e-8
#     vol = (vol - mean) / std
#     vol = np.clip(vol, -2.0, 3.0)
#     vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
#     return vol

# def _prepare_volume_for_show(x: np.ndarray):
#     vol = np.squeeze(np.array(x))
#     if vol.ndim == 4 and vol.shape[0] == 1:
#         vol = vol[0]
#     if vol.ndim == 3 and vol.shape[0] < vol.shape[-1]:
#         vol = np.moveaxis(vol, -1, 0)
#     return vol

# def save_volume_gif(volume_3d, title, index, is_image):
#     """Create a GIF for a single 3D volume (input, true, or predicted)."""
#     os.makedirs(SAVE_PATH, exist_ok=True)
#     volume_3d = _prepare_volume_for_show(volume_3d)

#     if is_image:
#         volume_3d = _normalize_image(volume_3d)
#         cmap, vmin, vmax = "plasma", 0.05, 0.95  # colorful MRI
#     else:
#         cmap, vmin, vmax = "tab10", 0, NUM_CLASSES - 1

#     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#     ims = []
#     step = max(1, volume_3d.shape[0] // 64)
#     slices = range(0, volume_3d.shape[0], step)

#     for i in slices:
#         frame = volume_3d[i, :, :]
#         im = ax.imshow(frame, cmap=cmap, vmin=vmin, vmax=vmax, animated=True)
#         if i == slices.start:
#             ax.set_title(title, fontsize=14)
#             ax.axis("off")
#         ims.append([im])

#     ani = animation.ArtistAnimation(fig, ims, interval=60, blit=True, repeat_delay=1000)
#     out_path = os.path.join(SAVE_PATH, f"{title}_{index}.gif")
#     try:
#         ani.save(out_path, writer="ffmpeg", fps=15, dpi=150)
#     except Exception:
#         ani.save(out_path, writer="pillow", fps=15)
#     plt.close(fig)
#     print(f"✅ Saved {title} animation to {out_path}")

# def save_combined_gif(img, true_mask, pred_mask, index):
#     """Side-by-side comparison (Input | True | Predicted) with colored MRI input."""
#     os.makedirs(SAVE_PATH, exist_ok=True)
#     img = _normalize_image(img)
#     img = _prepare_volume_for_show(img)
#     true_mask = _prepare_volume_for_show(true_mask)
#     pred_mask = _prepare_volume_for_show(pred_mask)

#     step = max(1, img.shape[0] // 64)
#     slices = range(0, img.shape[0], step)
#     fig, axes = plt.subplots(1, 3, figsize=(12, 5))
#     titles = ["Input", "True", "Predicted"]
#     ims = []

#     for a, t in zip(axes, titles):
#         a.set_title(t, fontsize=14)
#         a.axis("off")

#     for i in slices:
#         frame1 = axes[0].imshow(img[i], cmap="plasma", vmin=0.05, vmax=0.95, animated=True)
#         frame2 = axes[1].imshow(true_mask[i], cmap="tab10", vmin=0, vmax=NUM_CLASSES - 1, animated=True)
#         frame3 = axes[2].imshow(pred_mask[i], cmap="tab10", vmin=0, vmax=NUM_CLASSES - 1, animated=True)
#         ims.append([frame1, frame2, frame3])

#     ani = animation.ArtistAnimation(fig, ims, interval=60, blit=True, repeat_delay=1000)
#     out_path = os.path.join(SAVE_PATH, f"Combined_{index}.gif")
#     try:
#         ani.save(out_path, writer="ffmpeg", fps=15, dpi=150)
#     except Exception:
#         ani.save(out_path, writer="pillow", fps=15)
#     plt.close(fig)
#     print(f"✅ Saved combined animation to {out_path}")

# # -------------------------------
# # Core prediction
# # -------------------------------
# @torch.no_grad()
# def display_and_save_examples(dataset, number_of_examples, model, device):
#     model.eval()
#     loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

#     for count, (imgs, lbls) in enumerate(loader):
#         if lbls.ndim == 5 and lbls.shape[1] == 1:
#             lbls = lbls.squeeze(1)
#         imgs, lbls = imgs.to(device), lbls.long().to(device)

#         logits = model(imgs)
#         probs = F.softmax(logits, dim=1)
#         pred = probs.argmax(dim=1)

#         dpc = dice_per_class(logits, lbls)
#         mdc = float(dpc.mean())
#         print(f"\n🧩 Example {count+1}:")
#         print(f"  Multiclass Dice: {mdc:.4f}")
#         class_names = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]
#         for i, name in enumerate(class_names):
#             print(f"  {name} DSC: {dpc[i]:.4f}")

#         img_np = imgs.detach().cpu().numpy()[0, 0]
#         true_np = lbls.detach().cpu().numpy()[0]
#         pred_np = pred.detach().cpu().numpy()[0]

#         save_volume_gif(img_np, "InputImage", count, True)
#         save_volume_gif(true_np, "TrueMask", count, False)
#         save_volume_gif(pred_np, "PredictedMask", count, False)
#         save_combined_gif(img_np, true_np, pred_np, count)

#         if count + 1 >= number_of_examples:
#             break

# def predict_model():
#     print("🚀 Loading trained model...")
#     model_path = os.path.join(SAVED_RESULTS_PATH, "improved_3d_unet_model.pth")
#     model = ImprovedUNet3D().to(DEVICE)
#     model.load_state_dict(torch.load(model_path, map_location=DEVICE))
#     model.eval()
#     print(f"✅ Loaded model from {model_path}")

#     imgs, lbls = discover_pairs(DATA_ROOT_IMAGES, DATA_ROOT_LABELS)
#     print(f"✅ Found {len(imgs)} matching pairs between '{DATA_ROOT_IMAGES}' and '{DATA_ROOT_LABELS}'.")
#     test_dataset = Prostate3DDataset(imgs, lbls, downsample=(0.5, 0.5, 0.5), augment=False)
#     print(f"🧠 Dataset size: {len(test_dataset)} samples")

#     display_and_save_examples(test_dataset, 3, model, DEVICE)

# if __name__ == "__main__":
#     predict_model()


# """
# PyTorch prediction script for Improved 3D UNet model.
# Generates predictions and saves 3D animated GIFs for input, true, predicted,
# and combined (side-by-side) visualizations.

# Author: Nathan King (PyTorch port, enhanced visualization)
# """

# import os
# import numpy as np
# import torch
# import torch.nn.functional as F
# import matplotlib
# matplotlib.use("Agg")  # safe for remote/HPC
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from torch.utils.data import DataLoader

# from modules import ImprovedUNet3D
# from dataset import Prostate3DDataset, discover_pairs

# # -------------------------------
# # Import config from train.py
# # -------------------------------
# from train import (
#     SAVED_RESULTS_PATH,
#     DATA_ROOT_IMAGES,
#     DATA_ROOT_LABELS,
#     NUM_CLASSES,
# )

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SAVE_PATH = os.path.join(SAVED_RESULTS_PATH, "predictions")
# os.makedirs(SAVE_PATH, exist_ok=True)

# # -------------------------------
# # Metrics
# # -------------------------------
# def _to_one_hot_labels(lbls_long: torch.Tensor, num_classes: int) -> torch.Tensor:
#     oh = torch.nn.functional.one_hot(lbls_long, num_classes=num_classes)  # [B,D,H,W,C]
#     return oh.permute(0, 4, 1, 2, 3).float()  # [B,C,D,H,W]

# @torch.no_grad()
# def dice_per_class(pred_logits: torch.Tensor, gt_long: torch.Tensor, num_classes: int = NUM_CLASSES, eps: float = 1e-5):
#     probs = torch.softmax(pred_logits, dim=1)
#     pred_oh = torch.zeros_like(probs).scatter_(1, probs.argmax(dim=1, keepdim=True), 1)
#     gt_oh = _to_one_hot_labels(gt_long, num_classes)
#     dims = (0, 2, 3, 4)
#     inter = torch.sum(pred_oh * gt_oh, dim=dims)
#     denom = torch.sum(pred_oh + gt_oh, dim=dims)
#     dice_c = (2.0 * inter + eps) / (denom + eps)
#     return dice_c.detach().cpu().numpy()

# # -------------------------------
# # Visualization helpers
# # -------------------------------
# def _normalize_image(volume):
#     vol = np.array(volume, dtype=np.float32)
#     vmin, vmax = np.percentile(vol, (1, 99))
#     vol = np.clip((vol - vmin) / (vmax - vmin + 1e-8), 0, 1)
#     return vol

# def _prepare_volume_for_show(x: np.ndarray):
#     vol = np.squeeze(np.array(x))
#     if vol.ndim == 4 and vol.shape[0] == 1:
#         vol = vol[0]
#     if vol.ndim == 3 and vol.shape[0] < vol.shape[-1]:
#         vol = np.moveaxis(vol, -1, 0)
#     return vol

# def save_volume_gif(volume_3d, title, index, is_image):
#     """Individual GIF for one view (input, true, or predicted)."""
#     os.makedirs(SAVE_PATH, exist_ok=True)
#     if is_image:
#         volume_3d = _normalize_image(volume_3d)
#         cmap, vmin, vmax = "gray", 0, 1
#     else:
#         cmap, vmin, vmax = "tab10", 0, NUM_CLASSES - 1
#     volume_3d = _prepare_volume_for_show(volume_3d)

#     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#     ims = []
#     step = max(1, volume_3d.shape[0] // 64)
#     slices = range(0, volume_3d.shape[0], step)

#     for i in slices:
#         frame = volume_3d[i, :, :]
#         im = ax.imshow(frame, cmap=cmap, vmin=vmin, vmax=vmax, animated=True)
#         if i == slices.start:
#             ax.set_title(title, fontsize=14)
#             ax.axis("off")
#         ims.append([im])

#     ani = animation.ArtistAnimation(fig, ims, interval=60, blit=True, repeat_delay=1000)
#     out_path = os.path.join(SAVE_PATH, f"{title}_{index}.gif")
#     try:
#         ani.save(out_path, writer="ffmpeg", fps=15, dpi=150)
#     except Exception:
#         ani.save(out_path, writer="pillow", fps=15)
#     plt.close(fig)
#     print(f"✅ Saved {title} animation to {out_path}")

# def save_combined_gif(img, true_mask, pred_mask, index):
#     """Side-by-side comparison (Input | True | Predicted)."""
#     os.makedirs(SAVE_PATH, exist_ok=True)
#     img = _normalize_image(img)
#     img = _prepare_volume_for_show(img)
#     true_mask = _prepare_volume_for_show(true_mask)
#     pred_mask = _prepare_volume_for_show(pred_mask)

#     step = max(1, img.shape[0] // 64)
#     slices = range(0, img.shape[0], step)
#     fig, axes = plt.subplots(1, 3, figsize=(12, 5))
#     titles = ["Input", "True", "Predicted"]
#     ims = []

#     for a, t in zip(axes, titles):
#         a.set_title(t, fontsize=14)
#         a.axis("off")

#     for i in slices:
#         frame1 = axes[0].imshow(img[i], cmap="gray", vmin=0, vmax=1, animated=True)
#         frame2 = axes[1].imshow(true_mask[i], cmap="tab10", vmin=0, vmax=NUM_CLASSES-1, animated=True)
#         frame3 = axes[2].imshow(pred_mask[i], cmap="tab10", vmin=0, vmax=NUM_CLASSES-1, animated=True)
#         ims.append([frame1, frame2, frame3])

#     ani = animation.ArtistAnimation(fig, ims, interval=60, blit=True, repeat_delay=1000)
#     out_path = os.path.join(SAVE_PATH, f"Combined_{index}.gif")
#     try:
#         ani.save(out_path, writer="ffmpeg", fps=15, dpi=150)
#     except Exception:
#         ani.save(out_path, writer="pillow", fps=15)
#     plt.close(fig)
#     print(f"✅ Saved combined animation to {out_path}")

# # -------------------------------
# # Core prediction
# # -------------------------------
# @torch.no_grad()
# def display_and_save_examples(dataset, number_of_examples, model, device):
#     model.eval()
#     loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

#     for count, (imgs, lbls) in enumerate(loader):
#         if lbls.ndim == 5 and lbls.shape[1] == 1:
#             lbls = lbls.squeeze(1)
#         imgs, lbls = imgs.to(device), lbls.long().to(device)

#         logits = model(imgs)
#         probs = F.softmax(logits, dim=1)
#         pred = probs.argmax(dim=1)

#         dpc = dice_per_class(logits, lbls)
#         mdc = float(dpc.mean())
#         print(f"\n🧩 Example {count+1}:")
#         print(f"  Multiclass Dice: {mdc:.4f}")
#         class_names = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]
#         for i, name in enumerate(class_names):
#             print(f"  {name} DSC: {dpc[i]:.4f}")

#         img_np = imgs.detach().cpu().numpy()[0, 0]
#         true_np = lbls.detach().cpu().numpy()[0]
#         pred_np = pred.detach().cpu().numpy()[0]

#         save_volume_gif(img_np, "InputImage", count, True)
#         save_volume_gif(true_np, "TrueMask", count, False)
#         save_volume_gif(pred_np, "PredictedMask", count, False)
#         save_combined_gif(img_np, true_np, pred_np, count)

#         if count + 1 >= number_of_examples:
#             break

# def predict_model():
#     print("🚀 Loading trained model...")
#     model_path = os.path.join(SAVED_RESULTS_PATH, "improved_3d_unet_model.pth")
#     model = ImprovedUNet3D().to(DEVICE)
#     model.load_state_dict(torch.load(model_path, map_location=DEVICE))
#     model.eval()
#     print(f"✅ Loaded model from {model_path}")

#     imgs, lbls = discover_pairs(DATA_ROOT_IMAGES, DATA_ROOT_LABELS)
#     print(f"✅ Found {len(imgs)} matching pairs between '{DATA_ROOT_IMAGES}' and '{DATA_ROOT_LABELS}'.")
#     test_dataset = Prostate3DDataset(imgs, lbls, downsample=(0.5, 0.5, 0.5), augment=False)
#     print(f"🧠 Dataset size: {len(test_dataset)} samples")

#     display_and_save_examples(test_dataset, 3, model, DEVICE)

# if __name__ == "__main__":
#     predict_model()



# """
# PyTorch prediction script for Improved 3D UNet model.
# Generates predictions and saves 3D animated GIFs for input, true, and predicted masks
# with enhanced visualization quality.

# Author: Nathan King (PyTorch port, visualization improved)
# """

# import os
# import numpy as np
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from torch.utils.data import DataLoader

# from modules import ImprovedUNet3D
# from dataset import Prostate3DDataset, discover_pairs

# # -------------------------------
# # Import config from train.py (paths, constants)
# # -------------------------------
# from train import (
#     SAVED_RESULTS_PATH,
#     DATA_ROOT_IMAGES,
#     DATA_ROOT_LABELS,
#     NUM_CLASSES,
# )

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Where to save GIFs
# SAVE_PATH = os.path.join(SAVED_RESULTS_PATH, "predictions")
# os.makedirs(SAVE_PATH, exist_ok=True)

# # -------------------------------
# # Metrics
# # -------------------------------
# def _to_one_hot_labels(lbls_long: torch.Tensor, num_classes: int) -> torch.Tensor:
#     oh = torch.nn.functional.one_hot(lbls_long, num_classes=num_classes)  # [B,D,H,W,C]
#     return oh.permute(0, 4, 1, 2, 3).float()  # [B,C,D,H,W]

# @torch.no_grad()
# def dice_per_class(pred_logits: torch.Tensor, gt_long: torch.Tensor, num_classes: int = NUM_CLASSES, eps: float = 1e-5):
#     probs = torch.softmax(pred_logits, dim=1)  # [B,C,D,H,W]
#     pred_oh = torch.zeros_like(probs).scatter_(1, probs.argmax(dim=1, keepdim=True), 1)
#     gt_oh = _to_one_hot_labels(gt_long, num_classes)

#     dims = (0, 2, 3, 4)
#     inter = torch.sum(pred_oh * gt_oh, dim=dims)
#     denom = torch.sum(pred_oh + gt_oh, dim=dims)
#     dice_c = (2.0 * inter + eps) / (denom + eps)
#     return dice_c.detach().cpu().numpy()

# @torch.no_grad()
# def multiclass_dice(pred_logits: torch.Tensor, gt_long: torch.Tensor, num_classes: int = NUM_CLASSES, eps: float = 1e-5):
#     return float(dice_per_class(pred_logits, gt_long, num_classes, eps).mean())

# # -------------------------------
# # Visualization Helpers
# # -------------------------------
# def _prepare_volume_for_show(x: np.ndarray):
#     vol = np.array(x)
#     vol = np.squeeze(vol)
#     if vol.ndim == 3 and vol.shape[0] != vol.shape[1] and vol.shape[0] != vol.shape[2]:
#         vol = np.moveaxis(vol, -1, 0)  # ensure [D,H,W]
#     if vol.ndim == 4 and vol.shape[0] == 1:
#         vol = vol[0]
#     return vol  # [D,H,W]

# def save_volume_gif(volume_3d: np.ndarray, title: str, index: int, is_image: bool):
#     """
#     Save a 3D volume as an animated GIF (axial slices).
#     Enhanced version: better contrast, larger figure, smoother animation.
#     """
#     D = volume_3d.shape[0]

#     # Normalize contrast for images
#     if is_image:
#         vmin, vmax = np.percentile(volume_3d, 2), np.percentile(volume_3d, 98)
#     else:
#         vmin, vmax = 0, NUM_CLASSES - 1

#     fig, ax = plt.subplots(figsize=(6, 6))
#     ims = []

#     # Move axis so slices scroll correctly
#     vol = np.moveaxis(volume_3d, 0, -1)

#     for j in range(vol.shape[-1]):
#         frame = vol[..., j]
#         im = ax.imshow(
#             frame,
#             cmap="gray" if is_image else "tab10",
#             vmin=vmin,
#             vmax=vmax,
#             animated=True,
#         )
#         if j == 0:
#             ax.set_title(title, fontsize=14)
#             ax.axis("off")
#         ims.append([im])

#     ani = animation.ArtistAnimation(fig, ims, interval=80, blit=True, repeat_delay=1000)

#     out_path = os.path.join(SAVE_PATH, f"{title}_{index}.gif")
#     try:
#         ani.save(out_path, writer="ffmpeg", fps=15, dpi=150)
#     except Exception:
#         # fallback if ffmpeg not available
#         ani.save(out_path, writer="pillow", fps=15)
#     plt.close(fig)
#     print(f"✅ Saved {title} animation to {out_path}")

# # -------------------------------
# # Core prediction routine
# # -------------------------------
# @torch.no_grad()
# def display_and_save_examples(dataset, number_of_examples, model, device):
#     model.eval()
#     count = 0
#     loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

#     for batch in loader:
#         imgs, lbls = batch  # imgs: [B,1,D,H,W], lbls: [B,1,D,H,W] or [B,D,H,W]
#         if lbls.ndim == 5 and lbls.shape[1] == 1:
#             lbls = lbls.squeeze(1)  # [B,D,H,W]
#         imgs = imgs.to(device)
#         lbls = lbls.long().to(device)

#         logits = model(imgs)
#         probs = F.softmax(logits, dim=1)
#         pred = probs.argmax(dim=1)

#         dpc = dice_per_class(logits, lbls)
#         mdc = float(dpc.mean())

#         print(f"\n🧩 Example {count+1}:")
#         print(f"  Multiclass Dice: {mdc:.4f}")
#         class_names = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]
#         for i, name in enumerate(class_names):
#             print(f"  {name} DSC: {dpc[i]:.4f}")

#         img_np = imgs.detach().cpu().numpy()[0, 0]
#         true_np = lbls.detach().cpu().numpy()[0]
#         pred_np = pred.detach().cpu().numpy()[0]

#         img_np = _prepare_volume_for_show(img_np)
#         true_np = _prepare_volume_for_show(true_np)
#         pred_np = _prepare_volume_for_show(pred_np)

#         save_volume_gif(img_np, "InputImage", count, is_image=True)
#         save_volume_gif(true_np, "TrueMask", count, is_image=False)
#         save_volume_gif(pred_np, "PredictedMask", count, is_image=False)

#         count += 1
#         if count >= number_of_examples:
#             break

# def predict_model():
#     print("🚀 Loading trained model...")
#     model_path = os.path.join(SAVED_RESULTS_PATH, "improved_3d_unet_model.pth")

#     model = ImprovedUNet3D().to(DEVICE)
#     model.load_state_dict(torch.load(model_path, map_location=DEVICE))
#     model.eval()
#     print(f"✅ Loaded model from {model_path}")

#     imgs, lbls = discover_pairs(DATA_ROOT_IMAGES, DATA_ROOT_LABELS)
#     print(f"✅ Found {len(imgs)} matching pairs between '{DATA_ROOT_IMAGES}' and '{DATA_ROOT_LABELS}'.")
#     test_dataset = Prostate3DDataset(imgs, lbls, downsample=(0.5, 0.5, 0.5), augment=False)
#     print(f"🧠 Dataset size: {len(test_dataset)} samples")

#     display_and_save_examples(test_dataset, number_of_examples=3, model=model, device=DEVICE)

# if __name__ == "__main__":
#     predict_model()


# """
# PyTorch prediction script for Improved 3D UNet model.
# Generates predictions and saves 3D animated GIFs for input, true, and predicted masks.

# Author: Nathan King (PyTorch port)
# """

# import os
# import numpy as np
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from torch.utils.data import DataLoader

# from modules import ImprovedUNet3D
# from dataset import Prostate3DDataset, discover_pairs

# # -------------------------------
# # Import config from train.py (paths, constants)
# # -------------------------------
# # If you prefer to avoid importing, you can just paste the same constants here.
# from train import (
#     SAVED_RESULTS_PATH,
#     DATA_ROOT_IMAGES,
#     DATA_ROOT_LABELS,
#     NUM_CLASSES,
# )

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Where to save GIFs
# SAVE_PATH = os.path.join(SAVED_RESULTS_PATH, "predictions")
# os.makedirs(SAVE_PATH, exist_ok=True)

# # -------------------------------
# # Metrics (self-contained; no dependency on train.py)
# # -------------------------------
# def _to_one_hot_labels(lbls_long: torch.Tensor, num_classes: int) -> torch.Tensor:
#     # lbls_long: [B, D, H, W] integer labels
#     oh = torch.nn.functional.one_hot(lbls_long, num_classes=num_classes)  # [B, D, H, W, C]
#     return oh.permute(0, 4, 1, 2, 3).float()  # [B, C, D, H, W]

# @torch.no_grad()
# def dice_per_class(pred_logits: torch.Tensor, gt_long: torch.Tensor, num_classes: int = NUM_CLASSES, eps: float = 1e-5):
#     """
#     pred_logits: [B,C,D,H,W] (logits or probabilities)
#     gt_long:     [B,D,H,W]   (integer labels 0..C-1)
#     returns: (C,) dice per class, averaged over batch and spatial
#     """
#     probs = torch.softmax(pred_logits, dim=1)  # [B,C,D,H,W]
#     pred_oh = torch.zeros_like(probs).scatter_(1, probs.argmax(dim=1, keepdim=True), 1)  # one-hot argmax
#     gt_oh = _to_one_hot_labels(gt_long, num_classes)  # [B,C,D,H,W]

#     dims = (0, 2, 3, 4)
#     inter = torch.sum(pred_oh * gt_oh, dim=dims)
#     denom = torch.sum(pred_oh + gt_oh, dim=dims)
#     dice_c = (2.0 * inter + eps) / (denom + eps)  # [C]
#     return dice_c.detach().cpu().numpy()

# @torch.no_grad()
# def multiclass_dice(pred_logits: torch.Tensor, gt_long: torch.Tensor, num_classes: int = NUM_CLASSES, eps: float = 1e-5):
#     """Mean of per-class Dice (bounded in [0,1])."""
#     return float(dice_per_class(pred_logits, gt_long, num_classes, eps).mean())

# # -------------------------------
# # Visualization
# # -------------------------------
# def _prepare_volume_for_show(x: np.ndarray):
#     """
#     Accepts:
#       - image:       [D,H,W] OR [1,D,H,W] OR [H,W,D] OR [H,W,D,1]
#       - label/pred:  [D,H,W]
#     Returns [D,H,W].
#     """
#     vol = np.array(x)
#     # squeeze trailing or leading singletons
#     vol = np.squeeze(vol)
#     # If comes as [H,W,D], move D to axis 0
#     if vol.ndim == 3 and vol.shape[0] != vol.shape[1] and vol.shape[0] != vol.shape[2]:
#         # likely [H,W,D]
#         vol = np.moveaxis(vol, -1, 0)  # -> [D,H,W]
#     # If it's still [1,D,H,W], squeeze 0
#     if vol.ndim == 4 and vol.shape[0] == 1:
#         vol = vol[0]
#     return vol  # [D,H,W]

# def save_volume_gif(volume_3d: np.ndarray, title: str, index: int, is_image: bool):
#     """
#     volume_3d: [D,H,W]
#     """
#     fig, ax = plt.subplots()
#     ims = []
#     D = volume_3d.shape[0]

#     for j in range(D):
#         frame = volume_3d[j, :, :]
#         im = ax.imshow(
#             frame,
#             animated=True,
#             cmap="gray" if is_image else "tab10",
#             vmin=None if is_image else 0,
#             vmax=None if is_image else (NUM_CLASSES - 1),
#         )
#         if j == 0:
#             ax.imshow(
#                 frame,
#                 cmap="gray" if is_image else "tab10",
#                 vmin=None if is_image else 0,
#                 vmax=None if is_image else (NUM_CLASSES - 1),
#             )
#         ims.append([im])

#     ani = animation.ArtistAnimation(fig, ims, interval=60, blit=True, repeat_delay=1000)
#     out_path = os.path.join(SAVE_PATH, f"{title}_{index}.gif")
#     ani.save(out_path)
#     plt.close(fig)
#     print(f"✅ Saved {title} animation to {out_path}")

# # -------------------------------
# # Core prediction routine
# # -------------------------------
# @torch.no_grad()
# def display_and_save_examples(dataset, number_of_examples, model, device):
#     """
#     Iterate a few samples, run inference, print metrics, and save GIFs.
#     """
#     model.eval()
#     count = 0

#     loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

#     for batch in loader:
#         imgs, lbls = batch  # imgs: [B,1,D,H,W], lbls: [B,1,D,H,W] or [B,D,H,W]
#         # squeeze label channel if present
#         if lbls.ndim == 5 and lbls.shape[1] == 1:
#             lbls = lbls.squeeze(1)  # [B,D,H,W]
#         imgs = imgs.to(device)
#         lbls = lbls.long().to(device)

#         logits = model(imgs)  # [B,C,D,H,W]
#         probs  = F.softmax(logits, dim=1)
#         pred   = probs.argmax(dim=1, keepdim=False)  # [B,D,H,W]

#         # Metrics (per-sample)
#         dpc = dice_per_class(logits, lbls)  # (C,)
#         mdc = float(dpc.mean())

#         print(f"\n🧩 Example {count+1}:")
#         print(f"  Multiclass Dice: {mdc:.4f}")
#         class_names = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]
#         for i, name in enumerate(class_names):
#             print(f"  {name} DSC: {dpc[i]:.4f}")

#         # Prepare numpy for visualization
#         img_np  = imgs.detach().cpu().numpy()[0, 0]     # [D,H,W]
#         true_np = lbls.detach().cpu().numpy()[0]        # [D,H,W]
#         pred_np = pred.detach().cpu().numpy()[0]        # [D,H,W]

#         img_np  = _prepare_volume_for_show(img_np)
#         true_np = _prepare_volume_for_show(true_np)
#         pred_np = _prepare_volume_for_show(pred_np)

#         # Save GIFs
#         save_volume_gif(img_np,  "InputImage",   count, is_image=True)
#         save_volume_gif(true_np, "TrueMask",     count, is_image=False)
#         save_volume_gif(pred_np, "PredictedMask",count, is_image=False)

#         count += 1
#         if count >= number_of_examples:
#             break

# def predict_model():
#     """
#     Make predictions using the trained model and save animations.
#     """
#     print("🚀 Loading trained model...")
#     model_path = os.path.join(SAVED_RESULTS_PATH, "improved_3d_unet_model.pth")

#     model = ImprovedUNet3D().to(DEVICE)
#     model.load_state_dict(torch.load(model_path, map_location=DEVICE))
#     model.eval()
#     print(f"✅ Loaded model from {model_path}")

#     # Build dataset (same discovery and downsample as training; no augmentation for inference)
#     imgs, lbls = discover_pairs(DATA_ROOT_IMAGES, DATA_ROOT_LABELS)
#     print(f"✅ Found {len(imgs)} matching pairs between '{DATA_ROOT_IMAGES}' and '{DATA_ROOT_LABELS}'.")
#     test_dataset = Prostate3DDataset(imgs, lbls, downsample=(0.5, 0.5, 0.5), augment=False)
#     print(f"🧠 Dataset size: {len(test_dataset)} samples")

#     # Show 3 examples
#     display_and_save_examples(test_dataset, number_of_examples=3, model=model, device=DEVICE)

# if __name__ == "__main__":
#     predict_model()
