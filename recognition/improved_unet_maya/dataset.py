# dataset.py
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class OasisSliceDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transform=None):
        self.img_dir = Path(img_dir); self.lbl_dir = Path(lbl_dir)
        self.fnames = sorted([p.name for p in self.img_dir.glob("*.png")])
        self.transform = transform

    def __len__(self): return len(self.fnames)

    def __getitem__(self, idx):
        name = self.fnames[idx]
        img = np.array(Image.open(self.img_dir / name)).astype(np.float32)
        if img.ndim == 3: img = img[...,0]  # guard: grayscale
        # z-score normalisation (fallback to min-max if std==0)
        m, s = img.mean(), img.std()
        img = (img - m)/(s+1e-6)

        mask = np.array(Image.open(self.lbl_dir / name)).astype(np.int64)

        # torch tensors
        img = torch.from_numpy(img)[None, ...]       # [1,H,W]
        mask = torch.from_numpy(mask)                # [H,W]
        sample = {"image": img, "mask": mask, "name": name}

        if self.transform: sample = self.transform(sample)
        return sample
