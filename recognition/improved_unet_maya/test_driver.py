#!/usr/bin/env python3
"""
test_driver.py — COMP3710 test driver (UNet3D, HipMRI Prostate 3D)

- Loads the newest trained checkpoint from a persistent directory:
    /home/Student/s4740054/projects/final_project/models/improved_unet3d/unet3d/checkpoints/*.pt
  (override with UNET3D_MODEL_ROOT or --ckpt)

- Runs inference on HipMRI test set:
    /home/groups/comp3710/HipMRI_Study_open/{semantic_MRs,semantic_labels_only}
  (override with --mr_dir / --lbl_dir)

- Computes per-class & mean Dice; prints TEST_MEAN_DICE=... and
  saves metrics to:
    /home/Student/s4740054/projects/final_project/models/improved_unet3d/unet3d/test_driver_out/
  (override with --out_dir)
"""

import os, json, csv, argparse, glob
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import Prostate3DDataset
from modules import UNet3D

# ----------------- defaults aligned with your sbatch changes -----------------
PERSIST_ROOT = Path(os.environ.get(
    "UNET3D_MODEL_ROOT",
    "/home/Student/s4740054/projects/final_project/models/improved_unet3d"
))
DEFAULT_CKPT_DIR = PERSIST_ROOT / "unet3d" / "checkpoints"
DEFAULT_OUT_DIR  = PERSIST_ROOT / "unet3d" / "test_driver_out"

DEFAULT_MR_DIR  = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
DEFAULT_LBL_DIR = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"

# repo fallback if persistent area has no checkpoints
REPO_FALLBACK_CKPT_DIR = Path(__file__).resolve().parent / "runs" / "unet3d" / "checkpoints"

@torch.no_grad()
def dice_per_class_3d(logits, target, eps=1e-6):
    """logits: [B,C,D,H,W], target: [B,D,H,W] -> np.array [C]"""
    C = logits.shape[1]
    pred = torch.softmax(logits, dim=1)
    out = []
    for c in range(C):
        p = pred[:, c].reshape(-1)
        t = (target == c).float().reshape(-1)
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        out.append(((2 * inter + eps) / (denom + eps)).item())
    return np.array(out, dtype=float)

def find_checkpoint(ckpt_arg: str | None, ckpt_dir: Path) -> Path:
    """Resolve checkpoint path: explicit file > newest *.pt in persistent dir > newest in repo fallback."""
    # 1) explicit path
    if ckpt_arg:
        p = Path(ckpt_arg)
        if p.is_file():
            return p
        raise FileNotFoundError(f"--ckpt provided but not found: {p}")

    # 2) newest in persistent dir
    pts = sorted(glob.glob(str(ckpt_dir / "*.pt")), key=os.path.getmtime, reverse=True)
    if pts:
        return Path(pts[0])

    # 3) fallback: newest in repo runs/unet3d/checkpoints
    pts = sorted(glob.glob(str(REPO_FALLBACK_CKPT_DIR / "*.pt")), key=os.path.getmtime, reverse=True)
    if pts:
        return Path(pts[0])

    raise FileNotFoundError(
        f"No checkpoint found in:\n  {ckpt_dir}\n  {REPO_FALLBACK_CKPT_DIR}\n"
        "Tip: train first (sbatch unet3d_gpu_train.sbatch) or copy best_unet3d.pt to the checkpoints directory."
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mr_dir",  default=DEFAULT_MR_DIR)
    ap.add_argument("--lbl_dir", default=DEFAULT_LBL_DIR)
    ap.add_argument("--ckpt",    default=None, help="Path to a specific .pt file (optional)")
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve checkpoint path (persistent dir → fallback)
    ckpt_path = find_checkpoint(args.ckpt, DEFAULT_CKPT_DIR)
    print(f"[test_driver] Using checkpoint: {ckpt_path}")

    # Data
    ds = Prostate3DDataset(args.mr_dir, args.lbl_dir, strict=True)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # Model
    ckpt = torch.load(ckpt_path, map_location="cpu")
    n_classes = int(ckpt.get("n_classes", 5))
    model = UNet3D(in_channels=1, out_channels=n_classes, base=16).to(device).eval()
    model.load_state_dict(ckpt["model"])

    # Eval
    dices_all = []
    with torch.no_grad():
        for batch in dl:
            x = batch["image"].to(device).float()
            y = batch["mask"].to(device).long()
            logits = model(x)
            dices_all.append(dice_per_class_3d(logits, y))

    per_class = np.mean(np.stack(dices_all, axis=0), axis=0) if dices_all else np.zeros(n_classes)
    mean_dice = float(per_class.mean()) if per_class.size else float("nan")

    # Output (persistent)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "test_dice.json", "w") as f:
        json.dump({"per_class": per_class.tolist(), "mean": mean_dice}, f, indent=2)
    with open(out_dir / "test_dice.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_idx", "dice"])
        for i, d in enumerate(per_class):
            w.writerow([i, f"{d:.6f}"])
        w.writerow(["mean", f"{mean_dice:.6f}"])

    # Console print for marker/autograder
    print(f"[TEST] per-class Dice = {np.round(per_class, 3).tolist()}")
    print(f"TEST_MEAN_DICE={mean_dice:.6f}")
    print(f"[test_driver] Wrote: {out_dir/'test_dice.json'} and {out_dir/'test_dice.csv'}")

if __name__ == "__main__":
    main()
