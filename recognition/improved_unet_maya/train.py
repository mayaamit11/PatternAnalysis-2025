# train.py — train/validate/test CAN2D on OASIS with Dice+CE, plots & checkpoints
import argparse, os, math, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from modules import CAN2D
from dataset import OasisSliceDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv, os, time
##add ins 
from dataset import Prostate3DDataset
from modules import UNet3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#importing labels and the 
imgs = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
labs = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"


train_ds = Prostate3DDataset(imgs, labs, strict=True)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2)


model = UNet3D(in_channels=1, out_channels=5, base=16).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()


class CsvLogger:
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fieldnames)
        if new:
            self.w.writeheader()
            self.f.flush()
    def log(self, **kwargs):
        row = {k: kwargs.get(k) for k in self.fieldnames}
        row["timestamp"] = int(time.time())
        self.w.writerow(row)
        self.f.flush()
    def close(self):
        self.f.close()

def dice_per_class(logits, target, eps=1e-6):
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

class DiceCELoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
    def forward(self, logits, target):
        ce = self.ce(logits, target)
        C = logits.shape[1]
        pred = torch.softmax(logits, dim=1)
        tgt = torch.nn.functional.one_hot(target, C).permute(0,3,1,2).float()
        inter = (pred*tgt).sum(dim=(0,2,3))
        denom = pred.sum(dim=(0,2,3)) + tgt.sum(dim=(0,2,3))
        dice = (2*inter+1e-6)/(denom+1e-6)
        dice_loss = 1 - dice.mean()
        return 0.5*ce + 0.5*dice_loss

def plot_curves(log_path, history):
    os.makedirs(log_path, exist_ok=True)
    # Loss
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(log_path, "loss.png")); plt.close()
    # Dice
    plt.figure()
    plt.plot(history["val_mean_dice"], label="val mean dice")
    plt.xlabel("epoch"); plt.ylabel("dice"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(log_path, "val_dice.png")); plt.close()

def compute_n_classes(ds, probe_batches=3, batch_size=16):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    mx = 0
    for i, b in enumerate(dl):
        mx = max(mx, b["mask"].max().item())
        if i+1 >= probe_batches: break
    return mx + 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_imgs", default="/home/groups/comp3710/OASIS/keras_png_slices_train")
    ap.add_argument("--train_lbls", default="/home/groups/comp3710/OASIS/keras_png_slices_seg_train")
    ap.add_argument("--val_imgs",   default="/home/groups/comp3710/OASIS/keras_png_slices_validate")
    ap.add_argument("--val_lbls",   default="/home/groups/comp3710/OASIS/keras_png_slices_seg_validate")
    ap.add_argument("--test_imgs",  default="/home/groups/comp3710/OASIS/keras_png_slices_test")
    ap.add_argument("--test_lbls",  default="/home/groups/comp3710/OASIS/keras_png_slices_seg_test")
    ap.add_argument("--out_dir",    default="./runs")  # will create subfolder
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if (args.device=="auto" and torch.cuda.is_available()) else args.device)

    ds_tr = OasisSliceDataset(args.train_imgs, args.train_lbls)
    ds_va = OasisSliceDataset(args.val_imgs,   args.val_lbls)
    ds_te = OasisSliceDataset(args.test_imgs,  args.test_lbls)

    n_classes = compute_n_classes(ds_tr)
    model = CAN2D(n_classes=n_classes, base=args.base).to(device)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,  num_workers=4, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch*2, shuffle=False, num_workers=4, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=5, factor=0.5)
    loss_fn = DiceCELoss()

    use_amp = torch.cuda.is_available() and device.type=="cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run_dir = os.path.join(args.out_dir, "can2d")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir  = os.path.join(run_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True); os.makedirs(log_dir, exist_ok=True)

    best = -math.inf
    history = {"train_loss": [], "val_loss": [], "val_mean_dice": []}

    for epoch in range(1, args.epochs+1):
        # ---- train
        model.train()
        tr_losses = []
        for batch in dl_tr:
            x = batch["image"].to(device); y = batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tr_losses.append(loss.item())

        # ---- validate
        model.eval()
        va_losses, dices_all = [], []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            for batch in dl_va:
                x = batch["image"].to(device); y = batch["mask"].to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                va_losses.append(loss.item())
                dices_all.append(dice_per_class(logits, y).detach().cpu())
        val_loss = float(np.mean(va_losses)) if va_losses else float("inf")
        per_class = torch.cat(dices_all, dim=0).mean(0).numpy()
        mean_dice = float(per_class.mean())
        print(f"Epoch {epoch:03d}: train_loss={np.mean(tr_losses):.4f}  val_loss={val_loss:.4f}  "
              f"val_meanDice={mean_dice:.4f}  per-class={np.round(per_class,3)}")

        history["train_loss"].append(float(np.mean(tr_losses)))
        history["val_loss"].append(val_loss)
        history["val_mean_dice"].append(mean_dice)

        # step LR and save best
        sch.step(mean_dice)
        if mean_dice > best:
            best = mean_dice
            torch.save({"model": model.state_dict(),
                        "n_classes": n_classes,
                        "base": args.base}, os.path.join(ckpt_dir, "best_can2d.pt"))
            with open(os.path.join(ckpt_dir, "val_per_class.json"), "w") as f:
                json.dump({"per_class": per_class.tolist(), "mean": mean_dice}, f)

        # plot each epoch
        plot_curves(log_dir, history)

    # quick test at the end (same metric) — optional
    dl_te = DataLoader(ds_te, batch_size=args.batch*2, shuffle=False, num_workers=4, pin_memory=True)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best_can2d.pt"))["model"])
    model.eval(); dices_all = []
    with torch.no_grad():
        for batch in dl_te:
            x = batch["image"].to(device); y = batch["mask"].to(device)
            logits = model(x)
            dices_all.append(dice_per_class(logits, y).detach().cpu())
    per_class_te = torch.cat(dices_all).mean(0).numpy()
    print("TEST per-class Dice:", np.round(per_class_te, 3), "mean:", float(per_class_te.mean()))
    # optional assert if you want the run to fail if below target
    # assert (per_class_te >= 0.90).all(), f"Some classes below 0.90: {per_class_te}"

if __name__ == "__main__":
    main()
