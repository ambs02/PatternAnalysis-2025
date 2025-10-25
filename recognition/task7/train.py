"""
PyTorch training script for the 3D Improved UNet (Isensee et al. 2018).
Reproduces the TensorFlow version used in COMP3710 but uses torch + nibabel.
Includes Dice metrics, accuracy tracking, and plot saving.
"""

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
DATA_ROOT_IMAGES = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
DATA_ROOT_LABELS = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"

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

# -------------------------------
# Train & Validate
# -------------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, lbls.squeeze(1).long())


        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    dice_list = []
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        preds = model(imgs)
        loss = criterion(preds, lbls.squeeze(1).long())

        total_loss += loss.item()
        dice_list.append(dice_coefficient(preds, lbls))
    dice_array = np.array(dice_list)
    mean_dice = dice_array.mean(axis=0)
    return total_loss / len(loader), mean_dice


# -------------------------------
# Main Training Function
# -------------------------------
def main():
    print("🚀 Starting training of 3D Improved UNet...")
    imgs, lbls = discover_pairs(DATA_ROOT_IMAGES, DATA_ROOT_LABELS)
    print(f"✅ Found {len(imgs)} image/label pairs")

    dataset = Prostate3DDataset(imgs, lbls, downsample=(0.25, 0.25, 0.25), augment=True)
    n_total = len(dataset)
    n_test = int(TEST_SPLIT * n_total)
    n_val = int(VAL_SPLIT * n_total)
    n_train = n_total - n_test - n_val
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)
    test_loader = DataLoader(test_ds, batch_size=1)

    print(f"Dataset split: Train={n_train}, Val={n_val}, Test={n_test}")

    model = ImprovedUNet3D(num_classes=6).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses = [], []
    val_dice_scores = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, dice = validate(model, val_loader, criterion)
        t1 = time.time()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dice_scores.append(dice)

        print(f"Epoch {epoch}/{EPOCHS} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Dice: {dice.mean():.3f}, Time: {(t1 - t0):.1f}s")

    # -------------------------------
    # Save Results
    # -------------------------------
    torch.save(model.state_dict(), os.path.join(SAVE_PATH, "improved_3d_unet.pth"))
    print("✅ Model saved to", os.path.join(SAVE_PATH, "improved_3d_unet.pth"))

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(SAVE_PATH, "Loss.png"))
    plt.show()

    val_dice_scores = np.array(val_dice_scores)
    mean_dice_per_class = val_dice_scores.mean(axis=0)
    plt.figure()
    plt.bar(range(6), mean_dice_per_class)
    plt.title("Mean Dice Coefficient per Class (Validation)")
    plt.xlabel("Class")
    plt.ylabel("Dice")
    plt.savefig(os.path.join(SAVE_PATH, "Dice.png"))
    plt.show()

    print("✅ Mean Dice per class:", np.round(mean_dice_per_class, 3))

    # Evaluate on test set
    test_loss, test_dice = validate(model, test_loader, criterion)
    print("\n🧪 Test Results:")
    print("Test Loss:", round(test_loss, 4))
    print("Test Dice per class:", np.round(test_dice, 3))
    print("Mean Dice:", test_dice.mean().round(3))


if __name__ == "__main__":
    main()
