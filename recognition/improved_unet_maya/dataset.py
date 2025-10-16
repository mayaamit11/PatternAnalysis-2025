# dataset.py — OASIS PNG slice loader with robust image↔mask pairing
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

def _strip_double_ext(name: str):
    if name.endswith(".nii.png"):
        return name[:-len(".nii.png")], ".nii.png"
    if name.endswith(".png"):
        return name[:-len(".png")], ".png"
    return name, ""

def _mask_candidates_for_image(img_name: str):
    """
    Map image filename -> plausible mask filenames.
    Handles OASIS: case_086_slice_17(.nii).png -> seg_086_slice_17(.nii).png
    Tries plain .png + common *_seg/_mask fallbacks.
    """
    base, ext = _strip_double_ext(img_name)  # e.g., "case_086_slice_17"
    # Primary rule: 'case_' -> 'seg_'
    cand_core = base
    if base.startswith("case_"):
        cand_core = "seg_" + base[len("case_"):]
    elif base.startswith("img_"):
        cand_core = "seg_" + base[len("img_"):]
    elif base.startswith("seg_"):
        cand_core = base

    cands = []
    if ext: cands.append(f"{cand_core}{ext}")    # keep .nii.png if present
    cands.append(f"{cand_core}.png")             # many labels drop the .nii

    # generic fallbacks
    img_core, _ = _strip_double_ext(img_name)
    cands += [
        f"{img_core}.png",
        f"{img_core}_seg.png", f"{img_core}-seg.png",
        f"{img_core}_mask.png", f"{img_core}-mask.png",
    ]
    # de-dup
    seen, uniq = set(), []
    for c in cands:
        if c not in seen:
            uniq.append(c); seen.add(c)
    return uniq

class OasisSliceDataset(Dataset):
    """
    Returns dict: { "image": FloatTensor[1,H,W], "mask": LongTensor[H,W], "name": str }
    image: z-scored per-slice; mask: integer labels (0..K-1).
    """
    def __init__(self, img_dir, lbl_dir, transform=None, strict=True):
        self.img_dir = Path(img_dir); self.lbl_dir = Path(lbl_dir)
        self.transform = transform
        self.strict = strict

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
        if img.ndim == 3: img = img[..., 0]  # keep grayscale
        m, s = img.mean(), img.std()
        img = (img - m) / (s + 1e-6)         # z-score

        mask = np.array(Image.open(mask_path)).astype(np.int64)

        sample = {
            "image": torch.from_numpy(img)[None, ...],  # [1,H,W]
            "mask": torch.from_numpy(mask),             # [H,W]
            "name": img_path.name,
        }
        if self.transform: sample = self.transform(sample)
        return sample
