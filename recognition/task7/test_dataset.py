# test_dataset.py
"""
Test script for verifying dataset loading and preprocessing.
It:
  1. Loads MRI–label pairs using discover_pairs()
  2. Initializes the Prostate3DDataset
  3. Prints shapes and data types
  4. Displays one MRI slice with its segmentation overlay
"""

import torch
import matplotlib.pyplot as plt
from dataset import Prostate3DDataset, discover_pairs

# ====== Update your dataset paths here ======
images_root = "/Users/amberchen/Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-/data/HipMRI_study_complete_release_v1/semantic_MRs_anon"
labels_root = "/Users/amberchen/Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-/data/HipMRI_study_complete_release_v1/semantic_labels_anon"
# ============================================

# 1️⃣ Find MRI–label pairs
imgs, lbls = discover_pairs(images_root, labels_root)
print(f"✅ Found {len(imgs)} MRI volumes and {len(lbls)} label volumes")

# 2️⃣ Create dataset (test just a few samples)
dataset = Prostate3DDataset(
    imgs[:3], lbls[:3],
    downsample=(0.25, 0.25, 0.25),  # downsample by 4× (for speed)
    augment=False
)

# 3️⃣ Load one sample
img, lbl = dataset[0]
print("\n✅ Sample loaded successfully!")
print(f"Image shape: {tuple(img.shape)}")   # expected (1, D, H, W)
print(f"Label shape: {tuple(lbl.shape)}")   # expected (D, H, W) or (1, D, H, W)
print(f"Image dtype: {img.dtype}, Label dtype: {lbl.dtype}")

# Remove any leftover channel dimension from label if needed
lbl = lbl.squeeze(0) if lbl.ndim == 4 else lbl

# 4️⃣ Visualize a middle slice
D = img.shape[1]  # depth dimension
mid_slice = D // 2

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("MRI Slice")
plt.imshow(img[0, mid_slice].cpu(), cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Label Overlay")
plt.imshow(img[0, mid_slice].cpu(), cmap="gray")
plt.imshow(lbl[mid_slice].cpu(), alpha=0.4)
plt.axis("off")

plt.tight_layout()
plt.show()
