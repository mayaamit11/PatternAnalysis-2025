import argparse, os
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from dataset import OasisPNG2DPaired
from modules import UNet2D

def dice_coef(logits, target, eps=1e-6):
    p = (torch.sigmoid(logits) > 0.5).float()
    inter = (p * target).sum()
    denom = p.sum() + target.sum()
    return (2*inter + eps) / (denom + eps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_img_root", default="/home/groups/comp3710/OASIS/keras_png_slices_train")
    ap.add_argument("--train_msk_root", default="/home/groups/comp3710/OASIS/keras_png_slices_seg_train")
    ap.add_argument("--val_img_root",   default="/home/groups/comp3710/OASIS/keras_png_slices_validate")
    ap.add_argument("--val_msk_root",   default="/home/groups/comp3710/OASIS/keras_png_slices_seg_validate")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--outdir", default="/scratch/$USER/comp3710_models/improved_unet_maya")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.expandvars(args.outdir), exist_ok=True)

    train_ds = OasisPNG2DPaired(args.train_img_root, args.train_msk_root, img_size=args.img_size)
    val_ds   = OasisPNG2DPaired(args.val_img_root,   args.val_msk_root,   img_size=args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=4)

    # sanity check
    xb, yb = next(iter(train_loader))
    print("Batch shapes:", xb.shape, yb.shape)  # expect [B,1,256,256]

    model = UNet2D().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        tl, td, n = 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward(); opt.step()
            with torch.no_grad():
                bsz = x.size(0)
                tl += loss.item()*bsz
                td += dice_coef(logits, y).item()*bsz
                n += bsz
        tl /= n; td /= n

        model.eval()
        vl, vd, m = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                bsz = x.size(0)
                vl += loss.item()*bsz
                vd += dice_coef(logits, y).item()*bsz
                m += bsz
        vl /= m; vd /= m
        print(f"[{epoch}/{args.epochs}] train_loss={tl:.4f} dice={td:.4f} | val_loss={vl:.4f} dice={vd:.4f}")

        if vd > best:
            best = vd
            ckpt = os.path.expandvars(os.path.join(args.outdir, "unet2d_best.pt"))
            torch.save({"epoch": epoch, "state_dict": model.state_dict()}, ckpt)

if __name__ == "__main__":
    main()
