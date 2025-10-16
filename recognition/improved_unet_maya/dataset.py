import os, glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class OasisPNG2DPaired(Dataset):
    """
    Pair files by basename across two folders.
    Example:
      images_root = /home/groups/.../keras_png_slices_train
      masks_root  = /home/groups/.../keras_png_slices_seg_train
    """
    def __init__(self, images_root, masks_root, img_size=256):
        self.images_root = images_root
        self.masks_root = masks_root
        self.imgs = sorted(glob.glob(os.path.join(images_root, "*.png")))
        assert self.imgs, f"No PNGs found in {images_root}"
        self.msks = [os.path.join(masks_root, os.path.basename(p)) for p in self.imgs]
        for m in self.msks:
            if not os.path.exists(m):
                raise FileNotFoundError(f"Missing mask for {os.path.basename(m)}")
        self.img_size = img_size

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        x = Image.open(self.imgs[i]).convert("L")  # grayscale image
        y = Image.open(self.msks[i]).convert("L")  # grayscale mask (0/255)
        x = TF.to_tensor(x)                       # [1,H,W] in [0,1]
        y = (TF.to_tensor(y) > 0.5).float()       # binarise to {0,1}
        x = TF.resize(x, [self.img_size, self.img_size])
        y = TF.resize(y, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.NEAREST)
        return x, y
