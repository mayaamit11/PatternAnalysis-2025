# train.py — train/validate/test CAN2D on OASIS with Dice+CE, plots & checkpoints
import argparse, os, math, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from modules import CAN2D, UNet3D
from dataset import OasisSliceDataset, Prostate3DDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv, os, time
##add ins 


#checking the directory...
def ensure_dir(p):
    os.makedirs(p, exist_ok= True); return p

def plot_curves(log_path, history, tag):
    ensure_dir(log_path)
    # Loss
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    if len(history["val_loss"]) and not np.isnan(history["val_loss"][-1]):
        plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(log_path, f"{tag}_loss.png")); plt.close()
    # Dice
    if len(history["val_mean_dice"]):
        plt.figure()
        plt.plot(history["val_mean_dice"], label="val mean dice")
        plt.xlabel("epoch"); plt.ylabel("dice"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(log_path, f"{tag}_dice.png")); plt.close()

def compute_n_classes_from_loader(dl, num_batches=2):
    mx = 0
    for i, b in enumerate(dl):
        mx = max(mx, int(b["mask"].max().item()))
        if i+1 >= num_batches: break
    return mx + 1

#for the
def dice_per_class_2d(logits, target, eps=1e-6):
    # logits: [B,C,H,W], target: [B,H,W]
    C = logits.shape[1]
    pred = torch.softmax(logits, dim=1)
    dices = []
    for c in range(C):
        p = pred[:, c].reshape(-1)
        t = (target == c).float().reshape(-1)
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        dices.append((2 * inter + eps) / (denom + eps))
    return torch.stack(dices)  # [C]

def dice_per_class_3d(logits, target, eps=1e-6):
    # logits: [B,C,D,H,W], target: [B,D,H,W]
    C = logits.shape[1]
    pred = torch.softmax(logits, dim=1)
    dices = []
    for c in range(C):
        p = pred[:, c].reshape(-1)
        t = (target == c).float().reshape(-1)
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        dices.append((2 * inter + eps) / (denom + eps))
    return torch.stack(dices)


class DiceCELoss2D(nn.Module):
    def __init__(self, weight=None): super().__init__(); self.ce = nn.CrossEntropyLoss(weight=weight)
    def forward(self, logits, target):
        ce = self.ce(logits, target)
        C = logits.shape[1]
        pred = torch.softmax(logits, dim=1)
        tgt = torch.nn.functional.one_hot(target, C).permute(0,3,1,2).float()
        inter = (pred*tgt).sum(dim=(0,2,3))
        denom = pred.sum(dim=(0,2,3)) + tgt.sum(dim=(0,2,3))
        dice = (2*inter+1e-6)/(denom+1e-6)
        return 0.5*ce + 0.5*(1 - dice.mean())

class DiceCELoss3D(nn.Module):
    def __init__(self, weight=None): super().__init__(); self.ce = nn.CrossEntropyLoss(weight=weight)
    def forward(self, logits, target):
        ce = self.ce(logits, target)
        C = logits.shape[1]
        pred = torch.softmax(logits, dim=1)
        tgt = torch.nn.functional.one_hot(target, C).permute(0,4,1,2,3).float()
        inter = (pred*tgt).sum(dim=(0,2,3,4))
        denom = pred.sum(dim=(0,2,3,4)) + tgt.sum(dim=(0,2,3,4))
        dice = (2*inter+1e-6)/(denom+1e-6)
        return 0.5*ce + 0.5*(1 - dice.mean())


class CsvLogger:
    def __init__(self, path, fieldnames):
        self.path = path; self.fieldnames = fieldnames
        new = not os.path.exists(path)
        ensure_dir(os.path.dirname(path))
        self.f = open(path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fieldnames)
        if new: self.w.writeheader(); self.f.flush()
    def log(self, **kwargs):
        row = {k: kwargs.get(k) for k in self.fieldnames}
        row["timestamp"] = int(time.time())
        self.w.writerow(row); self.f.flush()
    def close(self): self.f.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["oasis2d","prostate3d"], default="oasis2d")

    # 2D OASIS defaults
    ap.add_argument("--train_imgs", default="/home/groups/comp3710/OASIS/keras_png_slices_train")
    ap.add_argument("--train_lbls", default="/home/groups/comp3710/OASIS/keras_png_slices_seg_train")
    ap.add_argument("--val_imgs",   default="/home/groups/comp3710/OASIS/keras_png_slices_validate")
    ap.add_argument("--val_lbls",   default="/home/groups/comp3710/OASIS/keras_png_slices_seg_validate")
    ap.add_argument("--test_imgs",  default="/home/groups/comp3710/OASIS/keras_png_slices_test")
    ap.add_argument("--test_lbls",  default="/home/groups/comp3710/OASIS/keras_png_slices_seg_test")

    # 3D Prostate defaults
    ap.add_argument("--mr_dir",  default="/home/groups/comp3710/HipMRI_Study_open/semantic_MRs")
    ap.add_argument("--lbl_dir", default="/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only")
    ap.add_argument("--crop", type=int, default=0, help="3D center-crop size, e.g. 160 (0 disables)")

    # shared
    ap.add_argument("--out_dir", default="./runs")
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)  # 2D only; 3D overrides to 1
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if (args.device=="auto" and torch.cuda.is_available()) else args.device)

    if args.task == "oasis2d":
        # -------- 2D: OASIS + CAN2D --------
        ds_tr = OasisSliceDataset(args.train_imgs, args.train_lbls)
        ds_va = OasisSliceDataset(args.val_imgs,   args.val_lbls)
        ds_te = OasisSliceDataset(args.test_imgs,  args.test_lbls)

        n_classes = compute_n_classes_from_loader(DataLoader(ds_tr, batch_size=32, shuffle=False))
        model = CAN2D(n_classes=n_classes, base=args.base).to(device)

        dl_tr = DataLoader(ds_tr, batch_size=args.batch,   shuffle=True,  num_workers=4, pin_memory=True)
        dl_va = DataLoader(ds_va, batch_size=args.batch*2, shuffle=False, num_workers=4, pin_memory=True)
        dl_te = DataLoader(ds_te, batch_size=args.batch*2, shuffle=False, num_workers=4, pin_memory=True)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=5, factor=0.5)
        loss_fn = DiceCELoss2D()

        run_dir = ensure_dir(os.path.join(args.out_dir, "can2d"))
        ckpt_dir, log_dir = ensure_dir(os.path.join(run_dir, "checkpoints")), ensure_dir(os.path.join(run_dir, "logs"))
        logger = CsvLogger(os.path.join(run_dir, "metrics.csv"), ["epoch","train_loss","val_loss","val_mean_dice"])

        best = -math.inf
        history = {"train_loss": [], "val_loss": [], "val_mean_dice": []}

        for epoch in range(1, args.epochs+1):
            # train
            model.train(); tr_losses=[]
            for b in dl_tr:
                x = b["image"].to(device); y = b["mask"].to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward(); opt.step()
                tr_losses.append(loss.item())

            # validate
            model.eval(); va_losses=[]; dices_all=[]
            with torch.no_grad():
                for b in dl_va:
                    x = b["image"].to(device); y = b["mask"].to(device)
                    logits = model(x)
                    va_losses.append(loss_fn(logits, y).item())
                    dices_all.append(dice_per_class_2d(logits, y).cpu())
            val_loss = float(np.mean(va_losses))
            per_class = torch.cat(dices_all).mean(0).numpy()
            mean_dice = float(per_class.mean())

            print(f"[2D] Epoch {epoch:03d} | train={np.mean(tr_losses):.4f}  val={val_loss:.4f}  dice={mean_dice:.4f}")
            history["train_loss"].append(float(np.mean(tr_losses)))
            history["val_loss"].append(val_loss)
            history["val_mean_dice"].append(mean_dice)
            logger.log(epoch=epoch, train_loss=np.mean(tr_losses), val_loss=val_loss, val_mean_dice=mean_dice)

            sch.step(mean_dice)
            plot_curves(log_dir, history, tag="2d")

            if mean_dice > best:
                best = mean_dice
                torch.save({"model": model.state_dict(),
                            "n_classes": n_classes,
                            "base": args.base},
                           os.path.join(ckpt_dir, "best_can2d.pt"))

        # quick test
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best_can2d.pt"))["model"])
        model.eval(); dices_all=[]
        with torch.no_grad():
            for b in dl_te:
                x = b["image"].to(device); y = b["mask"].to(device)
                logits = model(x)
                dices_all.append(dice_per_class_2d(logits, y).cpu())
        per_cls_te = torch.cat(dices_all).mean(0).numpy()
        print("TEST per-class Dice:", np.round(per_cls_te, 3), "mean:", float(per_cls_te.mean()))
        logger.close()

    else:

        # -------- 3D: Prostate + UNet3D --------
        full_ds = Prostate3DDataset(args.mr_dir, args.lbl_dir, strict=True)
        # 80/20 split
        # 80 goes into training and 20 into validating... 
        n_total = len(full_ds); n_tr = int(0.8*n_total); n_va = n_total - n_tr
        ds_tr, ds_va = random_split(full_ds, [n_tr, n_va], generator=torch.Generator().manual_seed(args.seed))

        # loaders (batch=1 for 3D)
        dl_tr = DataLoader(ds_tr, batch_size=1, shuffle=True,  num_workers=2)
        dl_va = DataLoader(ds_va, batch_size=1, shuffle=False, num_workers=2)

        n_classes = compute_n_classes_from_loader(dl_tr, num_batches=2)
        model = UNet3D(in_channels=1, out_channels=n_classes, base=max(16, args.base//4)).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = DiceCELoss3D()

        run_dir = ensure_dir(os.path.join(args.out_dir, "unet3d"))
        ckpt_dir, log_dir = ensure_dir(os.path.join(run_dir, "checkpoints")), ensure_dir(os.path.join(run_dir, "logs"))
        logger = CsvLogger(os.path.join(run_dir, "metrics.csv"), ["epoch","train_loss","val_loss","val_mean_dice"])

        def maybe_crop(x, y, c=args.crop):
            if c and c > 0:
                _,_,D,H,W = x.shape
                d0 = max((D-c)//2, 0); h0 = max((H-c)//2, 0); w0 = max((W-c)//2, 0)
                x = x[..., d0:d0+c, h0:h0+c, w0:w0+c]
                y = y[..., d0:d0+c, h0:h0+c, w0:w0+c]
            return x, y

        best = -math.inf
        history = {"train_loss": [], "val_loss": [], "val_mean_dice": []}

        for epoch in range(1, args.epochs+1):
            # train
            model.train(); tr_losses=[]
            for b in dl_tr:
                x = b["image"].to(device).float()  # [1,1,D,H,W]
                y = b["mask"].to(device).long()    # [1,D,H,W]
                x, y = maybe_crop(x, y)

                opt.zero_grad(set_to_none=True)
                logits = model(x)                  # [1,C,D,H,W]
                loss = loss_fn(logits, y)
                loss.backward(); opt.step()
                tr_losses.append(loss.item())

            # validate
            model.eval(); va_losses=[]; dices=[]
            with torch.no_grad():
                for b in dl_va:
                    x = b["image"].to(device).float()
                    y = b["mask"].to(device).long()
                    x, y = maybe_crop(x, y)
                    logits = model(x)
                    va_losses.append(loss_fn(logits, y).item())
                    dices.append(dice_per_class_3d(logits, y).cpu())
            val_loss = float(np.mean(va_losses)) if len(va_losses) else float("nan")
            per_cls = torch.stack(dices).mean(0).numpy() if len(dices) else np.array([])
            mean_dice = float(np.nanmean(per_cls)) if per_cls.size else float("nan")

            print(f"[3D] Epoch {epoch:03d} | train={np.mean(tr_losses):.4f}  val={val_loss:.4f}  dice_mean={mean_dice:.3f}")
            history["train_loss"].append(float(np.mean(tr_losses)))
            history["val_loss"].append(val_loss)
            history["val_mean_dice"].append(mean_dice)
            logger.log(epoch=epoch, train_loss=np.mean(tr_losses), val_loss=val_loss, val_mean_dice=mean_dice)

            plot_curves(log_dir, history, tag="3d")

            if not np.isnan(mean_dice) and mean_dice > best:
                best = mean_dice
                torch.save({"model": model.state_dict(),
                            "n_classes": n_classes},
                           os.path.join(ckpt_dir, "best_unet3d.pt"))
        logger.close()

if __name__ == "__main__":
    main() 