# predict.py — unified inference for OASIS 2D (CAN2D) and Prostate 3D (UNet3D)
import argparse, os, numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from dataset import OasisSliceDataset, Prostate3DDataset
from modules import CAN2D, UNet3D
import nibabel as nib

# a single entry-point for evaluating trained models:
#   - Task "oasis2d":  CAN2D on PNG slice pairs (OASIS)
#   - Task "prostate3d": UNet3D on 3D NIfTI volumes (HipMRI) <--- FINAL UPLOAD USING... 
#
# purpose:
#   * Load best checkpoint (or allow --random_init for pipeline tests)
#   * Run inference on test/held-out data
#   * Compute per-class Dice and mean Dice
#   * Save qualitative overlays (PNG) and (for 3D) NIfTI predictions
#
# Note: Keep logits → softmax/sigmoid inside metrics (not in model) to avoid
# numeric issues and to keep models reusable.


"Provides the inference and evaluation script. Loads the best trained checkpoint, runs prediction "
"on the test set, computes final Dice scores, and saves example "
"segmentation overlays for qualitative assessment."
# --- optional matplotlib (fallback to PIL if unavailable) ---
_HAS_MPL = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    _HAS_MPL = False

# --- palette for colorized masks ---
PALETTE = np.array([
    [0, 0, 0],       # 0 background
    [255, 0, 0],     # 1
    [0, 255, 0],     # 2
    [0, 0, 255],     # 3
    [255, 255, 0],   # 4 (prostate typically)
    [255, 0, 255],   # 5
    [0, 255, 255],   # 6
], dtype=np.uint8)

def name_to_mr_path(mr_dir, name):
    """Derive MRI path from a clean stem (handles optional suffixes)."""
    base = name
    # strip any suffixes if present
    for suf in ("_LFOV.nii.gz", ".nii.gz", ".nii", "_LFOV"):
        if base.endswith(suf):
            base = base[: -len(suf)]
    return os.path.join(mr_dir, f"{base}_LFOV.nii.gz")

def paint_image(mask_np: np.ndarray) -> Image.Image:
    """Convert a 2D integer mask to an RGB color image using PALETTE."""
    m = np.clip(mask_np, 0, len(PALETTE)-1)
    rgb = PALETTE[m]
    return Image.fromarray(rgb)


def dice_per_class_2d(logits, target, eps=1e-6):
    """
    Per-class Dice for 2D logits.
    logits: [B, C, H, W], target: [B, H, W] (integer labels)
    returns: [C] tensor (averaged over batch+pixels)
    """
    # logits: [B,C,H,W], target: [B,H,W]
    C = logits.shape[1]
    pred = torch.softmax(logits, dim=1)
    out = []
    for c in range(C):
        p = pred[:, c].reshape(-1)
        t = (target == c).float().reshape(-1)
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        out.append((2 * inter + eps) / (denom + eps))
    return torch.stack(out)


def dice_per_class_3d(logits, target, eps=1e-6):
    """
    Per-class Dice for 3D logits.
    logits: [B, C, D, H, W], target: [B, D, H, W] (integer labels)
    returns: [C] tensor (averaged over batch+voxels)
    """
    # logits: [B,C,D,H,W], target: [B,D,H,W]
    C = logits.shape[1]
    pred = torch.softmax(logits, dim=1)
    out = []
    for c in range(C):
        p = pred[:, c].reshape(-1)
        t = (target == c).float().reshape(-1)
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        out.append((2 * inter + eps) / (denom + eps))
    return torch.stack(out)


# -------- overlays with MPL (or PIL fallback) --------
def save_overlay2d(gray, mask, out_png):
    """Save side-by-side MRI and overlay for a single 2D slice."""
    if _HAS_MPL:
        plt.figure(figsize=(7,3))
        plt.subplot(1,2,1); plt.imshow(gray, cmap="gray"); plt.title("MRI")
        plt.subplot(1,2,2); plt.imshow(gray, cmap="gray")
        plt.imshow(mask, alpha=0.35); plt.title("Pred overlay")
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    else:
        g = gray
        g = (255*(g - g.min())/(g.ptp()+1e-6)).astype(np.uint8)
        g = Image.fromarray(g).convert("RGB")
        color = paint_image(mask.astype(np.int64))
        Image.blend(g, color, alpha=0.35).save(out_png)


def save_overlay3d(vol, mask, out_png_prefix):
    """
    Save a few central-ish slices (low/mid/high) as overlays for a 3D volume.
    vol, mask: [H, W, D] arrays
    """
    D = vol.shape[2]
    for frac, tag in [(0.4, "low"), (0.5, "mid"), (0.6, "high")]:
        k = int(D*frac)
        if _HAS_MPL:
            plt.figure(figsize=(7,3))
            plt.subplot(1,2,1); plt.imshow(vol[:,:,k], cmap="gray"); plt.title(f"Slice {k}")
            plt.subplot(1,2,2); plt.imshow(vol[:,:,k], cmap="gray")
            plt.imshow(mask[:,:,k], alpha=0.35); plt.title("Pred overlay")
            plt.tight_layout(); plt.savefig(f"{out_png_prefix}_{tag}.png", dpi=150); plt.close()
        else:
            g = vol[:,:,k]
            g = (g - g.mean())/(g.std()+1e-6)
            g = (255*(g - g.min())/(g.ptp()+1e-6)).astype(np.uint8)
            g = Image.fromarray(g).convert("RGB")
            color = paint_image(mask[:,:,k].astype(np.int64))
            Image.blend(g, color, alpha=0.35).save(f"{out_png_prefix}_{tag}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["oasis2d","prostate3d"], default="oasis2d")

    # 2D paths
    ap.add_argument("--test_imgs", default="/home/groups/comp3710/OASIS/keras_png_slices_test")
    ap.add_argument("--test_lbls", default="/home/groups/comp3710/OASIS/keras_png_slices_seg_test")
    ap.add_argument("--ckpt2d", default="./runs/can2d/checkpoints/best_can2d.pt")

    # 3D paths
    ap.add_argument("--mr_dir",  default="/home/groups/comp3710/HipMRI_Study_open/semantic_MRs")
    ap.add_argument("--lbl_dir", default="/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only")
    ap.add_argument("--ckpt3d", default="./runs/unet3d/checkpoints/best_unet3d.pt")
    ap.add_argument("--crop", type=int, default=0, help="center-crop size used at train time (set same to match)")

    # shared
    ap.add_argument("--out_dir", default="./runs/preds")
    ap.add_argument("--batch", type=int, default=32)  # 2D only; 3D uses batch=1
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--tta", action="store_true", help="flip TTA (2D); simple XYZ flips (3D)")

    # NEW: allow running without checkpoint to test the pipeline
    ap.add_argument("--random_init", action="store_true",
                    help="Skip checkpoint load; use random weights (for plumbing test)")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if (args.device=="auto" and torch.cuda.is_available()) else args.device)

    if args.task == "oasis2d":
        # ---------- 2D OASIS ----------
        ds_te = OasisSliceDataset(args.test_imgs, args.test_lbls)
        dl_te = DataLoader(ds_te, batch_size=args.batch, shuffle=False, num_workers=4)

        if args.random_init:
            # default to 5 classes for OASIS slices
            n_classes, base = 5, 64
            model = CAN2D(n_classes=n_classes, base=base).to(device).eval()
            print("[WARN] --random_init: 2D model using random weights; Dice will be meaningless.")
        else:
            ckpt = torch.load(args.ckpt2d, map_location="cpu")
            n_classes = ckpt.get("n_classes", 5)
            base = ckpt.get("base", 64)
            model = CAN2D(n_classes=n_classes, base=base).to(device).eval()
            model.load_state_dict(ckpt["model"])

        dices_all = []
        with torch.no_grad():
            for batch in dl_te:
                x = batch["image"].to(device)      # [B,1,H,W]
                y = batch["mask"].to(device)       # [B,H,W]

                if args.tta:
                    logits = model(x)
                    for dim in [2, 3]:  # H, W flips
                        logits += torch.flip(model(torch.flip(x, dims=(dim,))), dims=(dim,))
                    logits /= 3.0
                else:
                    logits = model(x)

                dices_all.append(dice_per_class_2d(logits, y).cpu())
                pred = torch.argmax(torch.softmax(logits, 1), 1)  # [B,H,W]

                # save a couple of example overlays
                for i in range(min(3, pred.shape[0])):
                    name = batch["name"][i].replace(".png","")
                    pm = pred[i].cpu().numpy().astype(np.int64)
                    paint_image(pm).save(os.path.join(args.out_dir, f"pred_{name}.png"))

        per_class = torch.cat(dices_all).mean(0).numpy()
        print("[2D] TEST per-class Dice:", np.round(per_class, 3), "mean:", float(per_class.mean()))

    else:
        # ---------- 3D Prostate ----------
        ds = Prostate3DDataset(args.mr_dir, args.lbl_dir, strict=True)
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

        if args.random_init:
            n_classes = 5  # HipMRI prostate labels commonly 0..4
            model = UNet3D(in_channels=1, out_channels=n_classes, base=16).to(device).eval()
            print("[WARN] --random_init: 3D model using random weights; Dice will be meaningless.")
        else:
            ckpt = torch.load(args.ckpt3d, map_location="cpu")
            n_classes = ckpt.get("n_classes", 5)
            model = UNet3D(in_channels=1, out_channels=n_classes, base=16).to(device).eval()
            model.load_state_dict(ckpt["model"])

        def maybe_crop(x, c=args.crop):
            if c and c > 0:
                _,_,D,H,W = x.shape
                d0 = max((D-c)//2, 0); h0 = max((H-c)//2, 0); w0 = max((W-c)//2, 0)
                return x[..., d0:d0+c, h0:h0+c, w0:w0+c], (d0,h0,w0), (D,H,W)
            return x, (0,0,0), (x.shape[-3], x.shape[-2], x.shape[-1])

        dices_all = []
        with torch.no_grad():
            for batch in dl:
                x = batch["image"].to(device).float()   # [1,1,D,H,W]
                y = batch["mask"].to(device).long()     # [1,D,H,W]
                name = batch["name"][0]                 # e.g. "B040_Week3"

                # reload MRI with affine via consistent name rule
                mr_path = name_to_mr_path(args.mr_dir, name)
                ni = nib.load(mr_path)
                affine = ni.affine
                vol_full = ni.get_fdata()

                x_crop, offsets, orig_shape = maybe_crop(x)
                if args.tta:
                    # simple XYZ flips TTA
                    logits = model(x_crop)
                    for dims in [(2,), (3,), (4,), (2,3), (2,4), (3,4), (2,3,4)]:
                        logits += torch.flip(model(torch.flip(x_crop, dims=dims)), dims=dims)
                    logits /= 8.0
                else:
                    logits = model(x_crop)              # [1,C,Dc,Hc,Wc]

                # Dice on the cropped region (aligned to crop)
                d0,h0,w0 = offsets
                Dc,Hc,Wc = x_crop.shape[-3:]
                target_crop = y[..., d0:d0+Dc, h0:h0+Hc, w0:w0+Wc]
                dices_all.append(dice_per_class_3d(logits, target_crop).cpu())

                # Argmax and un-crop to original volume shape
                pred_crop = torch.argmax(torch.softmax(logits, 1), 1).squeeze(0).cpu().numpy().astype(np.int16)
                D,H,W = orig_shape
                full_pred = np.zeros((D,H,W), dtype=np.int16)
                full_pred[d0:d0+Dc, h0:h0+Hc, w0:w0+Wc] = pred_crop

                # Write NIfTI prediction with same affine
                out_nii = os.path.join(args.out_dir, f"{name}_pred.nii.gz")
                nib.save(nib.Nifti1Image(full_pred, affine), out_nii)

                # Save overlays (central-ish slices)
                v = vol_full
                v = (v - v.mean())/(v.std() + 1e-6)
                out_png_prefix = os.path.join(args.out_dir, f"{name}")
                save_overlay3d(v, full_pred, out_png_prefix)

        if len(dices_all):
            per_class = torch.stack(dices_all).mean(0).numpy()
            print("[3D] TEST per-class Dice:", np.round(per_class, 3), "mean:", float(per_class.mean()))
        else:
            print("[3D] No samples processed.")


if __name__ == "__main__":
    main()
