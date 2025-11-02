import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---- Config ----
mri_path = "/root/COMP3710/assignment/PatternAnalysis-2025/recognition/improved_unet_maya/images/B006_Week0_LFOV.nii.gz"
lbl_path = "/root/COMP3710/assignment/PatternAnalysis-2025/recognition/improved_unet_maya/images/B006_Week0_SEMANTIC.nii.gz"
out_path = "/root/COMP3710/assignment/PatternAnalysis-2025/recognition/improved_unet_maya/metrics/example_gt_overlay.png"

# Choose which slice index to display
slice_idx = 64  # same as your model figure

# ---- Load volumes ----
mri = nib.load(mri_path).get_fdata()
lbl = nib.load(lbl_path).get_fdata()

# Clip index to valid range just in case
slice_idx = int(np.clip(slice_idx, 0, mri.shape[2]-1))

# Extract mid slice
img = mri[:, :, slice_idx].astype(np.float32)
mask = lbl[:, :, slice_idx].astype(np.int32)

# Normalize MRI for visibility
img = (img - img.min()) / (np.ptp(img) + 1e-6)

# Custom colormap for segmentation classes
cmap = ListedColormap([
    (0,0,0,0),      # background transparent
    (1,0,0,0.7),    # red
    (0,1,0,0.7),    # green
    (0,0,1,0.7),    # blue
    (1,0.84,0,0.7)  # gold (prostate)
])

# ---- Create figure ----
fig, axs = plt.subplots(1, 2, figsize=(10,5))

# Left: plain MRI
axs[0].imshow(img, cmap='gray')
axs[0].set_title(f"Slice {slice_idx}")
axs[0].axis('off')

# Right: MRI + GT overlay
axs[1].imshow(img, cmap='gray')
axs[1].imshow(np.ma.masked_where(mask == 0, mask), cmap=cmap, interpolation='nearest')
axs[1].set_title("Ground Truth overlay")
axs[1].axis('off')

plt.tight_layout()
plt.savefig(out_path, bbox_inches='tight', dpi=150)
plt.close()
print("✅ Saved overlay figure:", out_path)
