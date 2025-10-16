# train.py
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from dataset import OasisSliceDataset
from modules import ImprovedUNet

def dice_per_class(logits, target, eps=1e-6):
    # logits: [B,C,H,W], target: [B,H,W]
    C = logits.shape[1]
    pred = torch.softmax(logits, dim=1)
    dices = []
    for c in range(C):
        p = pred[:,c].contiguous().view(-1)
        t = (target==c).float().contiguous().view(-1)
        inter = (p*t).sum()
        denom = p.sum() + t.sum()
        dices.append((2*inter + eps)/(denom + eps))
    return torch.stack(dices)  # [C]

class DiceCELoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
    def forward(self, logits, target):
        ce = self.ce(logits, target)
        with torch.no_grad():
            pass
        # soft dice
        C = logits.shape[1]
        pred = torch.softmax(logits, dim=1)
        target_1h = torch.nn.functional.one_hot(target, C).permute(0,3,1,2).float()
        inter = (pred*target_1h).sum(dim=(0,2,3))
        denom = pred.sum(dim=(0,2,3)) + target_1h.sum(dim=(0,2,3))
        dice = (2*inter+1e-6)/(denom+1e-6)
        dice_loss = 1 - dice.mean()
        return 0.5*ce + 0.5*dice_loss

def main():
    train_imgs = "/home/groups/comp3710/OASIS/keras_png_slices_train"
    train_lbls = "/home/groups/comp3710/OASIS/keras_png_slices_seg_train"
    val_imgs   = "/home/groups/comp3710/OASIS/keras_png_slices_validate"
    val_lbls   = "/home/groups/comp3710/OASIS/keras_png_slices_seg_validate"

    ds_tr = OasisSliceDataset(train_imgs, train_lbls)
    ds_va = OasisSliceDataset(val_imgs,   val_lbls)
    dl_tr = DataLoader(ds_tr, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=32, shuffle=False, num_workers=4)

    n_classes =  (torch.stack([s['mask'] for s in [ds_tr[0]]]).max().item() + 1)
    model = ImprovedUNet(n_classes=n_classes).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=5, factor=0.5)
    loss_fn = DiceCELoss()

    scaler = torch.cuda.amp.GradScaler()
    best = 0.0
    for epoch in range(100):
        model.train()
        for batch in dl_tr:
            x = batch["image"].cuda(); y = batch["mask"].cuda()
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                out = model(x)
                if isinstance(out, tuple):  # deep supervision
                    main, a2, a1 = out
                    loss = loss_fn(main, y) + 0.3*loss_fn(a2, y) + 0.3*loss_fn(a1, y)
                else:
                    loss = loss_fn(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()

        # validate
        model.eval()
        dices = []
        with torch.no_grad():
            for batch in dl_va:
                x = batch["image"].cuda(); y = batch["mask"].cuda()
                logits = model(x)[0] if isinstance(model(x), tuple) else model(x)
                dices.append(dice_per_class(logits, y))
        mean_dice = torch.cat(dices).mean().item()
        sch.step(mean_dice)
        if mean_dice > best:
            best = mean_dice
            torch.save(model.state_dict(), "best_oasis_improved_unet.pt")
        print(f"Epoch {epoch}: val mean Dice={mean_dice:.4f}")

if __name__ == "__main__":
    main()
