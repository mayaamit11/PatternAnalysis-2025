# predict.py
import torch, numpy as np
from PIL import Image
from dataset import OasisSliceDataset
from modules import ImprovedUNet

def colorize(mask):
    # simple palette for up to 5 classes
    palette = np.array([[0,0,0],[255,0,0],[0,255,0],[0,0,255],[255,255,0]], dtype=np.uint8)
    return Image.fromarray(palette[mask].astype(np.uint8))

def run():
    test_imgs = "/home/groups/comp3710/OASIS/keras_png_slices_test"
    test_lbls = "/home/groups/comp3710/OASIS/keras_png_slices_seg_test"
    ds_te = OasisSliceDataset(test_imgs, test_lbls)
    model = ImprovedUNet(n_classes=5).cuda().eval()
    model.load_state_dict(torch.load("best_oasis_improved_unet.pt"))
    dices = []
    for i in range(len(ds_te)):
        s = ds_te[i]
        x = s["image"][None].cuda()
        with torch.no_grad():
            logits = model(x)
            if isinstance(logits, tuple): logits = logits[0]
            pred = torch.argmax(torch.softmax(logits,1),1)[0].cpu().numpy()
        # save pred
        colorize(pred).save(f"pred_{s['name']}")
    print("Done.")

if __name__ == "__main__":
    run()
