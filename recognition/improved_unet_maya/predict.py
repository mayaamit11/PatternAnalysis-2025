# predict.py — unified inference for OASIS 2D (CAN2D) and Prostate 3D (UNet3D)
import argparse, os, numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from dataset import OasisSliceDataset, Prostate3DDataset
from modules import CAN2D, UNet3D

# NIfTI library for 3D volumes
import nibabel as nib

# Use non-interactive backend (safe for SSH / HPC)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#colour pallette 
PALETTE = np.array([
    [0, 0, 0],       # 0 background
    [255, 0, 0],     # 1
    [0, 255, 0],     # 2
    [0, 0, 255],     # 3
    [255, 255, 0],   # 4 (prostate typically)
    [255, 0, 255],   # 5
    [0, 255, 255],   # 6
], dtype=np.uint8)



def paint_image(mask_np: np.ndarray) -> Image.Image:
    """Convert a 2D integer mask to an RGB color image using PALETTE."""
    m = np.clip(mask_np, 0, len(PALETTE)-1)
    rgb = PALETTE[m]
    return Image.fromarray(rgb)


###--------------------------------
def dice_per_class_2d(logits, target, eps=1e-6):
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
    # logits: [B,C,D,H,W], target: [B,D,H,W]
    C = logits.shape[1]
    pred = torch.softmax(logits, dim=1)
    out = []
    for c in range(C):
        p = pred[:, c].reshape(-1)
        t = (target == c).float().reshape(-1)
        inter = (p*t).sum()
        denom = p.sum() + t.sum()
        out.append((2*inter + eps)/(denom + eps))
    return torch.stack(out)


#----------------colour overalys
def save_overlay2d(gray, mask, out_png):
    plt.figure(figsize=(7,3))
    plt.subplot(1,2,1); plt.imshow(gray, cmap="gray"); plt.title("MRI")
    plt.subplot(1,2,2); plt.imshow(gray, cmap="gray")
    plt.imshow(mask, alpha=0.35); plt.title("Pred overlay")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def save_overlay3d(vol, mask, out_png_prefix):
    D = vol.shape[2]
    for frac, tag in [(0.4, "low"), (0.5, "mid"), (0.6, "high")]:
        k = int(D*frac)
        plt.figure(figsize=(7,3))
        plt.subplot(1,2,1); plt.imshow(vol[:,:,k], cmap="gray"); plt.title(f"Slice {k}")
        plt.subplot(1,2,2); plt.imshow(vol[:,:,k], cmap="gray")
        plt.imshow(mask[:,:,k], alpha=0.35); plt.title("Pred overlay")
        plt.tight_layout(); plt.savefig(f"{out_png_prefix}_{tag}.png", dpi=150); plt.close()

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

    ap.add_argument("--out_dir", default="./runs/preds")
    ap.add_argument("--batch", type=int, default=32)  # 2D only; 3D uses batch=1
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--tta", action="store_true", help="flip TTA (2D); simple XYZ flips (3D)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if (args.device=="auto" and torch.cuda.is_available()) else args.device)

    if args.task == "oasis2d":
        # ---------- 2D OASIS ----------
        ds_te = OasisSliceDataset(args.test_imgs, args.test_lbls)
        dl_te = DataLoader(ds_te, batch_size=args.batch, shuffle=False, num_workers=4)

        ckpt = torch.load(args.ckpt2d, map_location="cpu")
        n_classes = ckpt.get("n_classes", 5); base = ckpt.get("base", 64)

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

                # save a couple of example overlays from this batch
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
                name = batch["name"][0]

                # keep MRI & affine for writing
                # reload with nib to fetch affine/header (safer than storing in dataset)
                mr_path = os.path.join(args.mr_dir, f"{name}_LFOV.nii.gz")
                # fall back: use dataset’s underlying paired path if you’ve stored it; else reload via label path stem rule.
                # For generality, just re-find via ds logic:
                # (We know dataset paired label was stem+"_SEMANTIC.nii.gz", MRI was stem+"_LFOV.nii.gz")
                if mr_path is None:
                    # generic: try both suffixes
                    cand = os.path.join(args.mr_dir, f"{name}_LFOV.nii.gz")
                    mr_path = cand if os.path.exists(cand) else os.path.join(args.mr_dir, name + ".nii.gz")

                ni = nib.load(mr_path)
                affine = ni.affine
                vol_full = ni.get_fdata()  # for overlays (float)

                x_crop, offsets, orig_shape = maybe_crop(x)
                if args.tta:
                    # simple XYZ flips TTA
                    logits = model(x_crop)
                    for dims in [(2,), (3,), (4,), (2,3), (2,4), (3,4), (2,3,4)]:
                        logits += torch.flip(model(torch.flip(x_crop, dims=dims)), dims=dims)
                    logits /= 8.0
                else:
                    logits = model(x_crop)              # [1,C,Dc,Hc,Wc]

                # Dice
                dices_all.append(dice_per_class_3d(logits, y[..., offsets[0]:offsets[0]+x_crop.shape[-3],
                                                              offsets[1]:offsets[1]+x_crop.shape[-2],
                                                              offsets[2]:offsets[2]+x_crop.shape[-1]]).cpu())

                # Argmax and un-crop to original volume shape
                pred_crop = torch.argmax(torch.softmax(logits, 1), 1).squeeze(0).cpu().numpy().astype(np.int16)
                D,H,W = orig_shape
                full_pred = np.zeros((D,H,W), dtype=np.int16)
                d0,h0,w0 = offsets
                full_pred[d0:d0+pred_crop.shape[0], h0:h0+pred_crop.shape[1], w0:w0+pred_crop.shape[2]] = pred_crop

                # Write NIfTI prediction with same affine
                out_nii = os.path.join(args.out_dir, f"{name}_pred.nii.gz")
                nib.save(nib.Nifti1Image(full_pred, affine), out_nii)

                # Save a few overlays (central slices)
                out_png_prefix = os.path.join(args.out_dir, f"{name}")
                # rescale MRI for display
                v = vol_full
                v = (v - v.mean())/(v.std() + 1e-6)
                save_overlay3d(v, full_pred, out_png_prefix)

        if len(dices_all):
            per_class = torch.stack(dices_all).mean(0).numpy()
            print("[3D] TEST per-class Dice:", np.round(per_class, 3), "mean:", float(per_class.mean()))
        else:
            print("[3D] No samples processed.")

if __name__ == "__main__":
    main()