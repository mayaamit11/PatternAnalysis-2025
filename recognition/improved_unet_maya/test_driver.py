#!/usr/bin/env python3
"""
test_driver.py — COMP3710 test driver
Runs the final evaluation for the Improved UNet3D on the HipMRI dataset.
Loads the trained checkpoint, runs inference on the test set,
and prints the mean Dice coefficient.
"""

import os, torch, numpy as np
from torch.utils.data import DataLoader
from dataset import Prostate3DDataset
from modules import UNet3D
import nibabel as nib

def dice_per_class_3d(logits, target, eps=1e-6):
    """Compute per-class Dice for 3D volumes."""
    C = logits.shape[1]
    pred = torch.softmax(logits, dim=1)
    out = []
    for c in range(C):
        p = pred[:, c].reshape(-1)
        t = (target == c).float().reshape(-1)
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        out.append(((2 * inter + eps) / (denom + eps)).item())
    return np.array(out)

def main():
    # === Paths (marker can change if needed) ===
    mr_dir  = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
    lbl_dir = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"
    ckpt    = "runs/unet3d/checkpoints/best_unet3d.pt"
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load dataset ===
    ds = Prostate3DDataset(mr_dir, lbl_dir, strict=True)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    # === Load model & weights ===
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    ckpt_data = torch.load(ckpt, map_location="cpu")
    n_classes = int(ckpt_data.get("n_classes", 5))
    model = UNet3D(in_channels=1, out_channels=n_classes, base=16).to(device)
    model.load_state_dict(ckpt_data["model"])
    model.eval()

    # === Run evaluation ===
    dices = []
    with torch.no_grad():
        for batch in dl:
            x = batch["image"].to(device).float()  # [1,1,D,H,W]
            y = batch["mask"].to(device).long()    # [1,D,H,W]
            logits = model(x)
            dices.append(dice_per_class_3d(logits, y))
    per_class = np.mean(np.stack(dices), axis=0)
    mean_dice = float(per_class.mean())

    # === Print final metrics ===
    print(f"[TEST] per-class Dice = {np.round(per_class, 3).tolist()}")
    print(f"TEST_MEAN_DICE={mean_dice:.6f}")

if __name__ == "__main__":
    main()
