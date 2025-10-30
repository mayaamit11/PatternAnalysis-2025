# view_slice.py — simple MRI + segmentation viewer
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

# Always find files relative to where this script is
BASE = Path(__file__).resolve().parent

# MRI and mask paths
img_path = BASE / "B040_Week3_LFOV.nii.gz"
mask_path = BASE / "B040_Week3_SEMANTIC.nii.gz"

print("Loading:")
print(" -", img_path)
print(" -", mask_path)

# Load data
img = nib.load(str(img_path)).get_fdata()
mask = nib.load(str(mask_path)).get_fdata()

print("MRI shape:", img.shape)
print("Mask shape:", mask.shape)
print("Unique mask labels:", set(mask.flatten().astype(int)))

# Pick a middle slice
slice_idx = img.shape[2] // 2  # middle of z-axis
print(f"Showing slice {slice_idx}")

# Plot and save overlay
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img[:, :, slice_idx], cmap="gray")
plt.title(f"MRI Slice {slice_idx}")

plt.subplot(1, 2, 2)
plt.imshow(img[:, :, slice_idx], cmap="gray")
plt.imshow(mask[:, :, slice_idx], alpha=0.4)
plt.title("Overlay: Segmentation")

plt.tight_layout()
plt.savefig(BASE / "B040_Week3_overlay.png", dpi=150)
print("Saved:", BASE / "B040_Week3_overlay.png")
