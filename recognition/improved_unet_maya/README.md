# improved_unet_maya — OASIS 2D segmentation

- Data: PNG slices at `/home/groups/comp3710/OASIS/*`
- Model: UNet2D (binary), BCEWithLogits loss, Dice metric
- Image size: 256×256

## Train
```bash
python train.py --epochs 1
