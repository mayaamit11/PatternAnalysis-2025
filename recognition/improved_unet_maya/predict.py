import argparse, os, torch
import matplotlib.pyplot as plt
from dataset import OasisPNG2DPaired
from modules import UNet2D

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_img_root", default="/home/groups/comp3710/OASIS/keras_png_slices_validate")
    ap.add_argument("--val_msk_root", default="/home/groups/comp3710/OASIS/keras_png_slices_seg_validate")
    ap.add_argument("--ckpt", default="/scratch/$USER/comp3710_models/improved_unet_maya/unet2d_best.pt")
    ap.add_argument("--img_size", type=int, default=256)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = OasisPNG2DPaired(args.val_img_root, args.val_msk_root, img_size=args.img_size)
    x, y = ds[0]
    x_in = x.unsqueeze(0).to(device)

    model = UNet2D().to(device)
    state = torch.load(os.path.expandvars(args.ckpt), map_location=device)
    model.load_state_dict(state["state_dict"]); model.eval()

    with torch.no_grad():
        p = torch.sigmoid(model(x_in))[0,0].cpu()

    plt.figure(figsize=(9,3))
    plt.subplot(1,3,1); plt.imshow(x[0], cmap="gray"); plt.title("Image"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(y[0], cmap="gray"); plt.title("Mask");  plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(p,    cmap="gray"); plt.title("Pred");  plt.axis("off")
    os.makedirs("figures", exist_ok=True)
    out = "figures/pred_demo.png"
    plt.tight_layout(); plt.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
