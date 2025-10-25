"""
PyTorch training script for the 3D Improved UNet (Isensee et al. 2018).
Adds accuracy tracking and 5 training plots.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

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
NUM_CLASSES = 6


# -------------------------------
# Dice Coefficient
# -------------------------------
def dice_coefficient(pred, target, num_classes=NUM_CLASSES, epsilon=1e-5):
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
# Helper
# -------------------------------
def _prepare_targets(lbls: torch.Tensor) -> torch.Tensor:
    if lbls.ndim == 5 and lbls.shape[1] == 1:
        lbls = lbls.squeeze(1)
    return lbls.long()


# -------------------------------
# Training and Validation
# -------------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total_voxels = 0
    dice_scores_all = []

    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), _prepare_targets(lbls).to(DEVICE)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, lbls)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Accuracy
        preds_argmax = torch.argmax(preds, dim=1)
        correct += (preds_argmax == lbls).sum().item()
        total_voxels += lbls.numel()

        # Dice
        dice_scores_all.append(dice_coefficient(preds, lbls))

    train_acc = correct / total_voxels
    mean_dice_per_class = np.array(dice_scores_all).mean(axis=0)
    return total_loss / len(loader), train_acc, mean_dice_per_class


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_voxels = 0
    dice_scores_all = []

    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), _prepare_targets(lbls).to(DEVICE)
        preds = model(imgs)
        loss = criterion(preds, lbls)
        total_loss += loss.item()

        preds_argmax = torch.argmax(preds, dim=1)
        correct += (preds_argmax == lbls).sum().item()
        total_voxels += lbls.numel()

        dice_scores_all.append(dice_coefficient(preds, lbls))

    val_acc = correct / total_voxels
    mean_dice_per_class = np.array(dice_scores_all).mean(axis=0)
    return total_loss / len(loader), val_acc, mean_dice_per_class


# -------------------------------
# Main
# -------------------------------
def main():
    print("🚀 Starting 3D Improved UNet training...")

    imgs, lbls = discover_pairs(DATA_ROOT_IMAGES, DATA_ROOT_LABELS)
    print(f"✅ Found {len(imgs)} image/label pairs")

    dataset = Prostate3DDataset(imgs, lbls, downsample=(0.5, 0.5, 0.5), augment=True)
    n_total = len(dataset)
    n_test = int(TEST_SPLIT * n_total)
    n_val = int(VAL_SPLIT * n_total)
    n_train = n_total - n_test - n_val
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)
    test_loader = DataLoader(test_ds, batch_size=1)

    print(f"Dataset split: Train={n_train}, Val={n_val}, Test={n_test}")

    model = ImprovedUNet3D(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_dice_hist, val_dice_hist = [], []
    mean_dice_val = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc, train_dice = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_dice = validate(model, val_loader, criterion)
        t1 = time.time()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_dice_hist.append(train_dice)
        val_dice_hist.append(val_dice)
        mean_dice_val.append(val_dice.mean())

        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | "
              f"Mean Dice: {val_dice.mean():.3f} | Time: {(t1 - t0):.1f}s")

    # Save model
    torch.save(model.state_dict(), os.path.join(SAVE_PATH, "improved_3d_unet.pth"))

    epochs = np.arange(1, EPOCHS + 1)
    train_dice_hist = np.array(train_dice_hist)
    val_dice_hist = np.array(val_dice_hist)

    # -------------------------------
    # 1️⃣ Accuracy vs Epoch
    # -------------------------------
    plt.figure()
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.savefig(os.path.join(SAVE_PATH, "Accuracy_vs_Epoch.png"))
    plt.close()

    # -------------------------------
    # 2️⃣ Loss vs Epoch
    # -------------------------------
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.savefig(os.path.join(SAVE_PATH, "Loss_vs_Epoch.png"))
    plt.close()

    # -------------------------------
    # 3️⃣ Multiclass Mean Dice vs Epoch
    # -------------------------------
    plt.figure()
    plt.plot(epochs, mean_dice_val, label="Mean Dice (Validation)")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Coefficient")
    plt.title("Mean Dice Coefficient vs Epoch")
    plt.legend()
    plt.savefig(os.path.join(SAVE_PATH, "MeanDice_vs_Epoch.png"))
    plt.close()

    # -------------------------------
    # 4️⃣ Training Dice per Class
    # -------------------------------
    plt.figure()
    # for c in range(NUM_CLASSES):
    #     plt.plot(epochs, train_dice_hist[:, c], label=f"Class {c}")

    class_names = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]
    for c, name in enumerate(class_names):
        plt.plot(epochs, train_dice_hist[:, c], label=f"{name} DSC")

    plt.xlabel("Epoch")
    plt.ylabel("Dice Coefficient")
    plt.title("Training Dice Similarity Coefficient per Class vs Epoch")
    plt.legend()
    plt.savefig(os.path.join(SAVE_PATH, "TrainDice_perClass_vs_Epoch.png"))
    plt.close()

    # -------------------------------
    # 5️⃣ Validation Dice per Class
    # -------------------------------
    plt.figure()
    # for c in range(NUM_CLASSES):
    #     plt.plot(epochs, val_dice_hist[:, c], label=f"Class {c}")
    
    class_names = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]
    for c, name in enumerate(class_names):
        plt.plot(epochs, train_dice_hist[:, c], label=f"{name} DSC")
    
    plt.xlabel("Epoch")
    plt.ylabel("Dice Coefficient")
    plt.title("Validation Dice Similarity Coefficient per Class vs Epoch")
    plt.legend()
    plt.savefig(os.path.join(SAVE_PATH, "ValDice_perClass_vs_Epoch.png"))
    plt.close()

    # -------------------------------
    # 3b️ Multiclass Dice Coefficient vs Epoch
    # -------------------------------
    plt.figure()

    # Compute mean Dice (multiclass) for each epoch
    train_mean_dice = [d.mean() for d in train_dice_hist]
    val_mean_dice = [d.mean() for d in val_dice_hist]

    plt.plot(epochs, train_mean_dice, marker='o', label="Training Multiclass Dice Coefficient")
    plt.plot(epochs, val_mean_dice, marker='o', label="Validation Multiclass Dice Coefficient")

    # Add value labels on the validation curve for clarity
    for x, y in zip(epochs, val_mean_dice):
        plt.text(x, y - 0.05, f"{y:.3f}", ha="center", va="bottom", fontsize=8)

    plt.legend(loc="upper left")
    plt.title("Multiclass Dice Coefficient")
    plt.xlabel("Epoch")
    plt.ylabel("Multiclass Dice Coefficient")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, "MulticlassDiceCoefficient.png"))
    plt.close()


    print("✅ Saved 6 plots to ./results/")
    print("✅ Training complete.")



if __name__ == "__main__":
    main()


# """
# PyTorch training script for the 3D Improved UNet (Isensee et al. 2018).
# Reproduces the TensorFlow version used in COMP3710 but uses torch + nibabel.
# Includes Dice metrics, accuracy tracking, and plot saving.
# """

# import os
# import time
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
# import matplotlib.pyplot as plt
# import numpy as np

# from modules import ImprovedUNet3D
# from dataset import Prostate3DDataset, discover_pairs

# # -------------------------------
# # Config
# # -------------------------------
# DATA_ROOT_IMAGES = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
# DATA_ROOT_LABELS = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"

# SAVE_PATH = "./results/"
# os.makedirs(SAVE_PATH, exist_ok=True)

# EPOCHS = 10
# BATCH_SIZE = 2
# LR = 1e-4
# VAL_SPLIT = 0.1
# TEST_SPLIT = 0.1
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # -------------------------------
# # Dice Coefficient
# # -------------------------------
# def dice_coefficient(pred, target, num_classes=6, epsilon=1e-5):
#     """Multi-class Dice Similarity Coefficient"""
#     pred = torch.argmax(pred, dim=1)  # [B, D, H, W]
#     dice_scores = []
#     for cls in range(num_classes):
#         pred_cls = (pred == cls).float()
#         target_cls = (target == cls).float()
#         intersection = torch.sum(pred_cls * target_cls)
#         union = torch.sum(pred_cls) + torch.sum(target_cls)
#         dice = (2 * intersection + epsilon) / (union + epsilon)
#         dice_scores.append(dice.item())
#     return dice_scores

# # -------------------------------
# # Train & Validate
# # -------------------------------
# def train_one_epoch(model, loader, optimizer, criterion):
#     model.train()
#     total_loss = 0
#     for imgs, lbls in loader:
#         imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
#         optimizer.zero_grad()
#         preds = model(imgs)
#         loss = criterion(preds, lbls.squeeze(1).long())


#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     return total_loss / len(loader)


# @torch.no_grad()
# def validate(model, loader, criterion):
#     model.eval()
#     total_loss = 0
#     dice_list = []
#     for imgs, lbls in loader:
#         imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
#         preds = model(imgs)
#         loss = criterion(preds, lbls.squeeze(1).long())

#         total_loss += loss.item()
#         dice_list.append(dice_coefficient(preds, lbls))
#     dice_array = np.array(dice_list)
#     mean_dice = dice_array.mean(axis=0)
#     return total_loss / len(loader), mean_dice


# # -------------------------------
# # Main Training Function
# # -------------------------------
# def main():
#     print("🚀 Starting training of 3D Improved UNet...")
#     imgs, lbls = discover_pairs(DATA_ROOT_IMAGES, DATA_ROOT_LABELS)
#     print(f"✅ Found {len(imgs)} image/label pairs")

#     dataset = Prostate3DDataset(imgs, lbls, downsample=(0.25, 0.25, 0.25), augment=True)
#     n_total = len(dataset)
#     n_test = int(TEST_SPLIT * n_total)
#     n_val = int(VAL_SPLIT * n_total)
#     n_train = n_total - n_test - n_val
#     train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=1)
#     test_loader = DataLoader(test_ds, batch_size=1)

#     print(f"Dataset split: Train={n_train}, Val={n_val}, Test={n_test}")

#     model = ImprovedUNet3D(num_classes=6).to(DEVICE)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=LR)

#     train_losses, val_losses = [], []
#     val_dice_scores = []

#     for epoch in range(1, EPOCHS + 1):
#         t0 = time.time()
#         train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
#         val_loss, dice = validate(model, val_loader, criterion)
#         t1 = time.time()

#         train_losses.append(train_loss)
#         val_losses.append(val_loss)
#         val_dice_scores.append(dice)

#         print(f"Epoch {epoch}/{EPOCHS} - "
#               f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
#               f"Dice: {dice.mean():.3f}, Time: {(t1 - t0):.1f}s")

#     # -------------------------------
#     # Save Results
#     # -------------------------------
#     torch.save(model.state_dict(), os.path.join(SAVE_PATH, "improved_3d_unet.pth"))
#     print("✅ Model saved to", os.path.join(SAVE_PATH, "improved_3d_unet.pth"))

#     plt.figure()
#     plt.plot(train_losses, label="Train Loss")
#     plt.plot(val_losses, label="Val Loss")
#     plt.title("Loss vs Epochs")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.savefig(os.path.join(SAVE_PATH, "Loss.png"))
#     plt.show()

#     val_dice_scores = np.array(val_dice_scores)
#     mean_dice_per_class = val_dice_scores.mean(axis=0)
#     plt.figure()
#     plt.bar(range(6), mean_dice_per_class)
#     plt.title("Mean Dice Coefficient per Class (Validation)")
#     plt.xlabel("Class")
#     plt.ylabel("Dice")
#     plt.savefig(os.path.join(SAVE_PATH, "Dice.png"))
#     plt.show()

#     print("✅ Mean Dice per class:", np.round(mean_dice_per_class, 3))

#     # Evaluate on test set
#     test_loss, test_dice = validate(model, test_loader, criterion)
#     print("\n🧪 Test Results:")
#     print("Test Loss:", round(test_loss, 4))
#     print("Test Dice per class:", np.round(test_dice, 3))
#     print("Mean Dice:", test_dice.mean().round(3))


# if __name__ == "__main__":
#     main()
