import os, glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

try:
    import torchio as tio
except ImportError:
    tio = None

def discover_pairs(root_images, root_labels, pattern="*.nii*"):
    imgs = sorted(glob.glob(os.path.join(root_images, pattern)))
    lbls = sorted(glob.glob(os.path.join(root_labels, pattern)))
    assert len(imgs) == len(lbls) and len(imgs) > 0, "Image/label count mismatch"
    return imgs, lbls

class Prostate3DDataset(Dataset):
    """
    Returns:
      image: torch.FloatTensor [1, D, H, W]  (z-score normalized, downsampled)
      label: torch.LongTensor  [D, H, W]     (integer class ids)
    """
    def __init__(self, image_paths, label_paths,
                 downsample=(0.5, 0.5, 0.5), augment=False):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.downsample = downsample
        self.augment = augment

        if augment and tio is not None:
            self.tx = tio.Compose([
                tio.RandomFlip(axes=(0,1,2), flip_probability=0.5),
                tio.RandomAffine(scales=(0.9,1.1), degrees=10),
                tio.RandomElasticDeformation(num_control_points=5, max_displacement=5),
            ])
        else:
            self.tx = None

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, i):
        img = nib.load(self.image_paths[i]).get_fdata().astype(np.float32)   # (D,H,W)
        lbl = nib.load(self.label_paths[i]).get_fdata().astype(np.uint8)     # (D,H,W)

        # z-score normalization
        img = (img - img.mean()) / (img.std() + 1e-8)

        img_t = torch.from_numpy(img)[None,None]  # (1,1,D,H,W)
        lbl_t = torch.from_numpy(lbl)[None,None].float()

        # downsample (trilinear for image, nearest for label)
        img_t = F.interpolate(img_t, scale_factor=self.downsample,
                              mode='trilinear', align_corners=False)
        lbl_t = F.interpolate(lbl_t, scale_factor=self.downsample,
                              mode='nearest')

        img_t = img_t.squeeze(0)           # (1,D,H,W)
        lbl_t = lbl_t.squeeze(0).long()    # (D,H,W)

        if self.tx is not None:
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=img_t),
                label=tio.LabelMap(tensor=lbl_t)
            )
            subject = self.tx(subject)
            img_t, lbl_t = subject.image.tensor, subject.label.tensor

        return img_t, lbl_t


