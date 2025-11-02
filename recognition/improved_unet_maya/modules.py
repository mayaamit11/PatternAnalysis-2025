#modules.py

"Defines the Improved UNet3D architecture. Includes encoder and decoder blocks with 3D convolutions, "
"residual connections, and bottleneck dilation for improved context aggregation. "
"This file contains all model components as reusable PyTorch classes."

#  file contains reusable PyTorch modules for:
#   1) CAN2D: a dilated residual CNN for 2D segmentation (e.g., OASIS slices)
#   2) UNet3D: a compact 3D U-Net for volumetric segmentation (e.g., HipMRI)
#
# Notes:
# - Keep the heads (final conv layers) linear. Apply softmax/sigmoid in the
#   loss/metric code (train/predict) to avoid numerical issues.
# - Shapes:
#     CAN2D: input [B,1,H,W]  -> logits [B,C,H,W]
#     UNet3D: input [B,1,D,H,W] -> logits [B,C,D,H,W] <---- USING THIS 


import torch, torch.nn as nn, torch.nn.functional as F

def conv3(in_c, out_c, dilation=1):
    """
    3x3 2D convolution with 'same' padding relative to dilation.
    Bias is disabled; BatchNorm follows in blocks.
    """
    pad = dilation
    return nn.Conv2d(in_c, out_c, kernel_size=3, padding=pad, dilation=dilation, bias=False)


#convolutional block
def conv3d_block(in_c, out_c):
    """
    Two stacked 3x3x3 Conv3d + BN + ReLU blocks.
    Keeps spatial size when padding=1.
    """
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, 3, padding=1),
        nn.BatchNorm3d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_c, out_c, 3, padding=1),
        nn.BatchNorm3d(out_c),
        nn.ReLU(inplace=True),
    )


class CANBlock(nn.Module):
    """
    Dilated residual block for 2D feature aggregation.
    Structure: BN -> ReLU -> Conv(dilated 3x3) -> BN -> Conv(3x3) + residual.
    """
    def __init__(self, c, dilation):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c); self.conv1 = conv3(c, c, dilation)
        self.bn2 = nn.BatchNorm2d(c); self.conv2 = conv3(c, c, 1)
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)

class CAN2D(nn.Module):
    """
    Context Aggregation Network for 2D segmentation.

    Stem(3x3) -> Stack of dilated residual blocks (rates: 1,2,4,8,16,1)
               -> Dropout2d -> 1x1 classifier head

    Args:
        n_classes: number of output classes (channels of logits)
        base:     base channel width
        dropout:  dropout probability before the head
    """
    def __init__(self, n_classes, base=64, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base), nn.ReLU(inplace=True)
        )
        rates = [1, 2, 4, 8, 16, 1]
        self.blocks = nn.Sequential(*[CANBlock(base, r) for r in rates])
        self.dropout = nn.Dropout2d(dropout)    
        self.head = nn.Conv2d(base, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.dropout(x)
        return self.head(x)  # [B, C, H, W]


#unet added... 
class UNet3D(nn.Module):
    """
    Minimal 3D U-Net:
        enc1 -> pool -> enc2 -> pool -> enc3
        up2 + skip(enc2) -> dec2
        up1 + skip(enc1) -> dec1
        1x1x1 output head

    Args:
        in_channels:  input channels (e.g., 1 for MRI)
        out_channels: classes (logit channels)
        base:         base feature width (scales as base, 2*base, 4*base)
    """
    def __init__(self, in_channels=1, out_channels=5, base=16):
        super().__init__()
        self.enc1 = conv3d_block(in_channels, base)
        self.pool = nn.MaxPool3d(2)
        self.enc2 = conv3d_block(base, base*2)
        self.enc3 = conv3d_block(base*2, base*4)

        self.up2 = nn.ConvTranspose3d(base*4, base*2, 2, 2)
        self.dec2 = conv3d_block(base*4, base*2)
        self.up1 = nn.ConvTranspose3d(base*2, base, 2, 2)
        self.dec1 = conv3d_block(base*2, base)

        self.outc = nn.Conv3d(base, out_channels, 1)

    
    def forward(self, x):
        """
        x: [B, in_channels, D, H, W]
        return: logits [B, out_channels, D, H, W]
        """
        e1 = self.enc1(x)          # B,base,D,H,W
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], 1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], 1)
        d1 = self.dec1(d1)
        return self.outc(d1)