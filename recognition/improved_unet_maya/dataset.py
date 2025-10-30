# dataset.py — OASIS (2D) + Prostate (3D) loaders

from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib   # <-- needed for NIfTI
import os, glob, random

# ---------- OASIS 2D (unchanged) for the first question ----------
def _strip_double_ext(name: str):
    if name.endswith(".nii.png"):
        return name[:-len(".nii.png")], ".nii.png"
    if name.endswith(".png"):
        return name[:-len(".png")], ".png"
    return name, ""

def _mask_candidates_for_image(img_name: str):
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

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
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
    Loads paired 3D MRI volumes and segmentation masks (HipMRI Study, Prostate 3D).
    Returns:
      image: FloatTensor [1, D, H, W]
      mask:  LongTensor  [D, H, W]
      name:  str (stem)
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
        img_path, mask_path = self.pairs[idx]
        img_ni  = nib.load(img_path)
        mask_ni = nib.load(mask_path) #the mask is the 

        # canonical orientation (prevents RAS/LPS mismatches)
        img  = nib.as_closest_canonical(img_ni).get_fdata().astype(np.float32)
        mask = nib.as_closest_canonical(mask_ni).get_fdata().astype(np.int64)

        if img.shape != mask.shape:
            # final guard; if one has an extra singleton dim, fix here
            if img.ndim == 4 and img.shape[-1] == 1: img = img[...,0]
            if mask.ndim == 4 and mask.shape[-1] == 1: mask = mask[...,0]
            if img.shape != mask.shape:
                raise RuntimeError(f"Image/label shape mismatch: {img.shape} vs {mask.shape} for {img_path.name}")

        # z-score per volume
        m, s = img.mean(), img.std()
        img = (img - m) / (s + 1e-6)

        # channel-first for 3D convs
        img = np.expand_dims(img, 0)  # [1, D, H, W]

        sample = {
            "image": torch.from_numpy(img).float(),  # float32
            "mask":  torch.from_numpy(mask).long(),  # class indices
            "name":  img_path.stem,
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
