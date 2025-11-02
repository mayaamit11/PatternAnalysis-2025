# dataset.py — HipMRI Prostate 3D loader for UNet3D


# Provides a PyTorch Dataset for volumetric (3D) MRI segmentation on the
# HipMRI Study (Prostate). Pairs *_LFOV.nii.gz MRI volumes with their
# *_SEMANTIC.nii.gz label volumes, normalises the MRI, and returns tensors
# shaped for 3D UNet:
#   image: FloatTensor [1, D, H, W]
#   mask:  LongTensor  [D, H, W]
#
# Example:
#   ds = Prostate3DDataset(
#       image_dir="/home/groups/comp3710/HipMRI_Study_open/semantic_MRs",
#       label_dir="/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only",
#   )
#   sample = ds[0]
#   sample["image"].shape -> [1, D, H, W]
#   sample["mask"].shape  -> [D, H, W]

from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib   # <-- needed for NIfTI
import os, glob, random

# ---------- OASIS 2D (unchanged) for the first question ----------
def _strip_double_ext(name: str):
    #get rid of double extensions like '.nii.png' or single '.png'
    if name.endswith(".nii.png"):
        return name[:-len(".nii.png")], ".nii.png"
    if name.endswith(".png"):
        return name[:-len(".png")], ".png"
    return name, ""

def _mask_candidates_for_image(img_name: str):
    """
    create all possible segmentation mask name candidates
    given an image file name (e.g., case_01.png -> seg_01.png).
    Ensures correct pairing between images and their marked layered masks.
    """

    base, ext = _strip_double_ext(img_name)
    cand_core = base
    if base.startswith("case_"):
        cand_core = "seg_" + base[len("case_"):]
    elif base.startswith("img_"):
        cand_core = "seg_" + base[len("img_"):]
    elif base.startswith("seg_"):
        cand_core = base

    cands = []
    if ext: cands.append(f"{cand_core}{ext}")
    cands.append(f"{cand_core}.png")

    img_core, _ = _strip_double_ext(img_name)
    cands += [
        f"{img_core}.png",
        f"{img_core}_seg.png", f"{img_core}-seg.png",
        f"{img_core}_mask.png", f"{img_core}-mask.png",
    ]
    seen, uniq = set(), []
    for c in cands:
        if c not in seen:
            uniq.append(c); seen.add(c)
    return uniq

class OasisSliceDataset(Dataset):
    """
    Dataset for 2D OASIS slices.
    Pairs .png images and their corresponding segmentation masks.
    Performs z-score normalisation and returns tensor.
    """

    def __init__(self, img_dir, lbl_dir, transform=None, strict=True):
        self.img_dir = Path(img_dir); self.lbl_dir = Path(lbl_dir)
        self.transform = transform; self.strict = strict

        imgs = sorted(self.img_dir.glob("*.png"))
        self.pairs = []; missing = []
        for ip in imgs:
            found = None
            for cand in _mask_candidates_for_image(ip.name):
                mp = self.lbl_dir / cand
                if mp.exists():
                    found = mp; break
            if found is None:
                missing.append(ip.name)
            else:
                self.pairs.append((ip, found))
        if missing and strict:
            ex = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"No matching masks for {len(missing)} image(s). E.g. {ex}. Check {self.lbl_dir}"
            )

    def __len__(self): 
        return len(self.pairs)


    def __getitem__(self, idx):
        """
        retruns one sample containing:
          image: [1, H, W] FloatTensor
          mask:  [H, W]   LongTensor
          name:  str      image name
        """
        img_path, mask_path = self.pairs[idx]
        img = np.array(Image.open(img_path)).astype(np.float32)
        if img.ndim == 3: img = img[..., 0]
        m, s = img.mean(), img.std()
        img = (img - m) / (s + 1e-6)
        mask = np.array(Image.open(mask_path)).astype(np.int64)

        sample = {
            "image": torch.from_numpy(img)[None, ...],   # [1,H,W]
            "mask":  torch.from_numpy(mask).long(),      # [H,W]
            "name":  img_path.name,
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

# ---------- Prostate 3D (Task 7) ----------
##example data
class Prostate3DDataset(Dataset):
    """
        Dataset for 3D HipMRI Prostate volumes.
        Loads paired 3D MRI (.nii.gz) and segmentation mask files.

        Each sample contains:
        image: FloatTensor [1, D, H, W]
        mask:  LongTensor  [D, H, W]
        name:  string (volume identifier)

        Example usage:
            ds = Prostate3DDataset(
                image_dir="/home/groups/comp3710/HipMRI_Study_open/semantic_MRs",
                label_dir="/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"
            )
        """

    #this allows us to train later... using an overlay of the img_dir and the label 
    # train_ds = Prostate3DDataset(img_dir, label_dir)
    #basically if you go into the image section, it displays a slice 
    # of a black and white png and an overlay that has been annotated (in colour)
    # we are trainig the model to recognise the prostate region on a scan alike to a human (prostate in gold)


    def __init__(self, image_dir, label_dir, transform=None, strict=True, limit=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.strict = strict

        # inside Prostate3DDataset.__init__
        imgs = sorted(Path(image_dir).glob("*_LFOV.nii.gz"))
        self.pairs, missing = [], []
        for ip in imgs:
            stem = ip.name.replace("_LFOV.nii.gz", "")   # e.g. "B006_Week0"
            mp = Path(label_dir) / f"{stem}_SEMANTIC.nii.gz"
            if mp.exists():
                self.pairs.append((ip, mp))
            else:
                missing.append(ip.name)
        if missing and strict:
            raise FileNotFoundError(f"Missing labels for e.g. {missing[:3]}")

        if limit:
            self.pairs = self.pairs[:limit]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Load one 3D volume and its mask.
        Returns a dictionary suitable for training or evaluation.
        """
        img_path, mask_path = self.pairs[idx]
        img = nib.load(img_path).get_fdata().astype(np.float32)
        mask = nib.load(mask_path).get_fdata().astype(np.int64)

        # ... (normalization, expand_dims, etc) ...

        # ✅ clean stem: e.g. "B006_Week0"
        clean_name = img_path.name.replace("_LFOV.nii.gz", "")

        sample = {
            "image": torch.from_numpy(img[None, ...]).float(),  # [1,D,H,W]
            "mask":  torch.from_numpy(mask).long(),             # [D,H,W]
            "name":  clean_name,                                # <- use clean name
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
