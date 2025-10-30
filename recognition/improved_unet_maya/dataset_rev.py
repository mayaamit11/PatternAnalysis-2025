# ---------- 3D Prostate dataset (append to dataset.py) ----------
import os, glob, random
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

def _zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-8)

def _sample_crop(img, lbl, size, pos_weight=0.7):
    """
    img: (C, D, H, W), lbl: (D, H, W), size: (d, h, w)
    Bias crops to include foreground with prob pos_weight.
    """
    C, D, H, W = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    d, h, w = size
    def rs(DX, sx): return 0 if DX <= sx else random.randint(0, DX - sx)
    if (lbl > 0).any() and random.random() < pos_weight:
        zs, ys, xs = np.where(lbl > 0)
        cz, cy, cx = random.choice(list(zip(zs, ys, xs)))
        sd = np.clip(cz - d // 2, 0, max(0, D - d))
        sh = np.clip(cy - h // 2, 0, max(0, H - h))
        sw = np.clip(cx - w // 2, 0, max(0, W - w))
    else:
        sd, sh, sw = rs(D, d), rs(H, h), rs(W, w)
    return img[:, sd:sd+d, sh:sh+h, sw:sw+w], lbl[sd:sd+d, sh:sh+h, sw:sw+w]

class Prostate3DDataset(Dataset):
    """
    Expects MSD-like tree:
      root/
        imagesTr/case_0001_0000.nii.gz  [required]
                 case_0001_0001.nii.gz  [optional second modality]
        labelsTr/case_0001.nii.gz
    """
    def __init__(self, root, split="train", patch_size=(128,128,64), augment=True):
        super().__init__()
        self.root = root
        self.patch = patch_size
        self.augment = augment

        img_dir = os.path.join(root, "imagesTr" if split in ("train","val") else "imagesTs")
        lbl_dir = os.path.join(root, "labelsTr")
        ids = sorted({os.path.basename(p).split("_")[0] for p in glob.glob(os.path.join(img_dir, "*.nii.gz"))})

        items = []
        for cid in ids:
            ch0 = os.path.join(img_dir, f"{cid}_0000.nii.gz")
            ch1 = os.path.join(img_dir, f"{cid}_0001.nii.gz")
            lbl = os.path.join(lbl_dir, f"{cid}.nii.gz")
            if not (os.path.exists(ch0) and os.path.exists(lbl)):
                continue
            chans = [ch0] + ([ch1] if os.path.exists(ch1) else [])
            items.append((chans, lbl))

        # simple split: last 10% as val
        n = len(items)
        if split == "train":
            self.items = items[: int(0.9 * n)]
        elif split == "val":
            self.items = items[int(0.9 * n):]
        else:
            self.items = items  # test

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        chan_paths, lbl_path = self.items[i]
        # label first (for targeted crop)
        lbl = nib.load(lbl_path).get_fdata(caching='unchanged').astype(np.int64)
        # load channels
        imgs = []
        for p in chan_paths:
            x = nib.load(p).get_fdata(caching='unchanged')
            imgs.append(_zscore(x))
        img = np.stack(imgs, axis=0)  # (C, D, H, W)

        # light augments: flips
        if self.augment:
            if random.random() < 0.5: img = img[:, ::-1, :, :]; lbl = lbl[::-1, :, :]
            if random.random() < 0.5: img = img[:, :, ::-1, :]; lbl = lbl[:, ::-1, :]
            if random.random() < 0.5: img = img[:, :, :, ::-1]; lbl = lbl[:, :, ::-1]

        # crop
        if self.patch is not None:
            img, lbl = _sample_crop(img, lbl, self.patch, pos_weight=0.7)

        return torch.from_numpy(img.copy()).float(), torch.from_numpy(lbl.copy()).long()
