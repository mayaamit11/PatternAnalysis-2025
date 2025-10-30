# predict.py — load best checkpoint, run on test set, report Dice & save overlays
import argparse, os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from modules import CAN2D
from dataset import OasisSliceDataset

PALETTE = np.array([
    [0, 0, 0],       # 0 background
    [255, 0, 0],     # 1
    [0, 255, 0],     # 2
    [0, 0, 255],     # 3
    [255, 255, 0],   # 4 (prostate typically)
    [255, 0, 255],   # 5
    [0, 255, 255],   # 6
], dtype=np.uint8)


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

def colorize(mask):
    c = PALETTE[np.clip(mask, 0, len(PALETTE)-1)]
    return Image.fromarray(c)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_imgs", default="/home/groups/comp3710/OASIS/keras_png_slices_test")
    ap.add_argument("--test_lbls", default="/home/groups/comp3710/OASIS/keras_png_slices_seg_test")
    ap.add_argument("--ckpt", default="./runs/can2d/checkpoints/best_can2d.pt")
    ap.add_argument("--out_dir", default="./runs/can2d/preds")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--tta", action="store_true", help="flip TTA")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if (args.device=="auto" and torch.cuda.is_available()) else args.device)

    ds_te = OasisSliceDataset(args.test_imgs, args.test_lbls)
    dl_te = torch.utils.data.DataLoader(ds_te, batch_size=args.batch, shuffle=False, num_workers=4)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    n_classes = ckpt.get("n_classes", 5)
    base = ckpt.get("base", 64)

    model = CAN2D(n_classes=n_classes, base=base).to(device).eval()
    model.load_state_dict(ckpt["model"])

    dices_all = []
    with torch.no_grad():
        for batch in dl_te:
            x = batch["image"].to(device); y = batch["mask"].to(device)
            if args.tta:
                # avg softmax over original + H/V flips
                logits = model(x)
                for dim in [2, 3]:  # H then W flips
                    x_flip = torch.flip(x, dims=(dim,))
                    logits += torch.flip(model(x_flip), dims=(dim,))
                logits /= 3.0
            else:
                logits = model(x)

            dices_all.append(dice_per_class(logits, y).detach().cpu())
            pred = torch.argmax(torch.softmax(logits, 1), 1)  # [B,H,W]

            # save a few overlays for the first batch
            for i in range(min(3, pred.shape[0])):
                name = batch["name"][i]
                pm = pred[i].cpu().numpy().astype(np.int64)
                colorize(pm).save(os.path.join(args.out_dir, f"pred_{name}"))

    per_class = torch.cat(dices_all).mean(0).numpy()
    print("TEST per-class Dice:", np.round(per_class, 3), "mean:", float(per_class.mean()))
    # enforce goal if desired:
    # assert (per_class >= 0.90).all(), f"Fail: {per_class}"

if __name__ == "__main__":
    main()
