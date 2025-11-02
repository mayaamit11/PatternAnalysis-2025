#!/usr/bin/env python3
"""
test_driver.py — COMP3710 test driver (Improved UNet3D, HipMRI Prostate Segmentation)

Runs final evaluation of the trained Improved UNet3D model on the HipMRI dataset.
- Loads trained checkpoint automatically (if present)
- Runs 3D inference on all volumes
- Computes per-class and mean Dice
- Saves metrics to persistent folder under models/improved_unet3d/unet3d/test_driver_out
- Works out of the box on Rangpur with no manual folder setup
"""

import os, json, csv, argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import Prostate3DDataset
from modules import UNet3D

# ----------------- Directory setup -----------------
# Base persistent model directory (relative to repo root)
PERSIST_ROOT = Path(os.environ.get(
    "UNET3D_MODEL_ROOT",
    Path(__file__).resolve().parents[2] / "models" / "improved_unet3d"
))
CKPT_PATH = PERSIST_ROOT / "unet3d" / "checkpoints" / "best_unet3d.pt"
OUT_DIR   = PERSIST_ROOT / "unet3d" / "test_driver_out"
CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Default dataset paths (Rangpur)
DEFAULT_MR_DIR  = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
DEFAULT_LBL_DIR = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"


# ----------------- Dice metric -----------------
@torch.no_grad()
def dice_per_class_3d(logits, target, eps=1e-6):
    """
    logits: [B,C,D,H,W], target: [B,D,H,W]
    returns: np.array [C] per-class dice
    """
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


# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mr_dir",  default=DEFAULT_MR_DIR)
    ap.add_argument("--lbl_dir", default=DEFAULT_LBL_DIR)
    ap.add_argument("--ckpt",    default=str(CKPT_PATH))
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--crop", type=int, default=160, help="center crop size (set 0 to disable)")
    ap.add_argument("--tta", action="store_true", help="enable flip-based test-time augmentation")
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load checkpoint ---
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Ensure you have trained the model or copied your checkpoint."
        )
    print(f"[test_driver] Using checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    n_classes = int(ckpt.get("n_classes", 5))

    # --- Dataset & loader ---
    ds = Prostate3DDataset(args.mr_dir, args.lbl_dir, strict=True)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # --- Model ---
    model = UNet3D(in_channels=1, out_channels=n_classes, base=16).to(device).eval()
    model.load_state_dict(ckpt["model"])

    # --- Crop helper ---
    def maybe_crop(x, c=args.crop):
        if c and c > 0:
            _, _, D, H, W = x.shape
            d0 = max((D - c) // 2, 0)
            h0 = max((H - c) // 2, 0)
            w0 = max((W - c) // 2, 0)
            return x[..., d0:d0 + c, h0:h0 + c, w0:w0 + c], (d0, h0, w0), (D, H, W)
        return x, (0, 0, 0), x.shape[-3:]

    # --- Evaluation loop ---
    dices_all = []
    with torch.no_grad():
        for batch in dl:
            x = batch["image"].to(device).float()  # [1,1,D,H,W]
            y = batch["mask"].to(device).long()    # [1,D,H,W]
            x_crop, (d0, h0, w0), (D, H, W) = maybe_crop(x)

            # Optional test-time augmentation (flip-TTA)
            if args.tta:
                logits = model(x_crop)
                for dims in [(2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]:
                    logits += torch.flip(model(torch.flip(x_crop, dims=dims)), dims=dims)
                logits /= 8.0
            else:
                logits = model(x_crop)

            target_crop = y[..., d0:d0 + x_crop.shape[-3],
                             h0:h0 + x_crop.shape[-2],
                             w0:w0 + x_crop.shape[-1]]
            dices_all.append(dice_per_class_3d(logits, target_crop))

    # --- Aggregate metrics ---
    per_class = np.mean(np.stack(dices_all, axis=0), axis=0) if dices_all else np.zeros(n_classes)
    mean_dice = float(per_class.mean()) if per_class.size else float("nan")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Save results ---
    with open(out_dir / "test_dice.json", "w") as f:
        json.dump({"per_class": per_class.tolist(), "mean": mean_dice}, f, indent=2)

    with open(out_dir / "test_dice.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_idx", "dice"])
        for i, d in enumerate(per_class):
            w.writerow([i, f"{d:.6f}"])
        w.writerow(["mean", f"{mean_dice:.6f}"])

    # --- Print summary for autograder ---
    print(f"[TEST] per-class Dice = {np.round(per_class, 3).tolist()}")
    print(f"TEST_MEAN_DICE={mean_dice:.6f}")
    print(f"[test_driver] Wrote: {out_dir / 'test_dice.json'} and {out_dir / 'test_dice.csv'}")


if __name__ == "__main__":
    main()
