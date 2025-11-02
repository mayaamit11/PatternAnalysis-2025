import os, nibabel as nib, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- paths (adjust case_id if needed) ---
case_id = "B006_Week0"
base = "/root/COMP3710/assignment/PatternAnalysis-2025/recognition/improved_unet_maya"
img_path = f"{base}/images/{case_id}_LFOV.nii.gz"
lbl_path = f"{base}/images/{case_id}_SEMANTIC.nii.gz"
out_dir  = f"{base}/metrics"
os.makedirs(out_dir, exist_ok=True)

# --- choose the exact slice index you want to show ---
SLICE_IDX = 64   # match your model figure

# --- load volumes ---
img = nib.load(img_path).get_fdata()
lbl = nib.load(lbl_path).get_fdata()

# guard
SLICE_IDX = int(np.clip(SLICE_IDX, 0, img.shape[2]-1))

# --- extract slice ---
im = img[:, :, SLICE_IDX].astype(np.float32)
gt = lbl[:, :, SLICE_IDX].astype(np.int32)

# normalize MRI to 0..1 for viewing
im = (im - im.min()) / (np.ptp(im) + 1e-6)


# 1) save the clean input slice
plt.imsave(f"{out_dir}/example_input.png", im, cmap="gray")

# 2) save GT overlay (transparent bg + distinct class colors)
#    0=bg transparent; 1..4 colored (edit if your classes differ)
cmap = ListedColormap([
    (0,0,0,0.0),       # background transparent
    (1,0,0,1.0),       # class 1 red
    (0,1,0,1.0),       # class 2 green
    (0,0,1,1.0),       # class 3 blue
    (1,0.84,0,1.0),    # class 4 gold (prostate)
])
plt.figure(figsize=(6,6))
plt.imshow(im, cmap="gray")
plt.imshow(np.ma.masked_where(gt==0, gt), cmap=cmap, alpha=0.35, interpolation="nearest")
plt.axis("off"); plt.tight_layout()
plt.savefig(f"{out_dir}/example_gt.png", bbox_inches="tight", pad_inches=0, dpi=150)
plt.close()

print("Wrote:", f"{out_dir}/example_input.png", f"{out_dir}/example_gt.png")
