# modules.py
import torch, torch.nn as nn, torch.nn.functional as F

def conv3x3(in_c, out_c): return nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = conv3x3(c, c); self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = conv3x3(c, c); self.bn2 = nn.BatchNorm2d(c)
    def forward(self, x):
        id = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + id)

class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            ResBlock(out_c), nn.Dropout2d(0.1),
        )
    def forward(self, x): return self.seq(x)

class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.conv = nn.Sequential(
            conv3x3(out_c*2, out_c), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            ResBlock(out_c), nn.Dropout2d(0.1),
        )
    def forward(self, x, skip):
        x = self.up(x)                # -> [B, out_c, H*, W*]
        x = torch.cat([x, skip], 1)   # concat with skip of same out_c
        return self.conv(x)           # keeps channels at out_c

class ImprovedUNet(nn.Module):
    def __init__(self, n_classes, base=32, deep_supervision=True):
        super().__init__()
        self.deep = deep_supervision

        # encoder
        self.stem = nn.Sequential(
            nn.Conv2d(1, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base), nn.ReLU(inplace=True), ResBlock(base)
        )
        self.d1 = Down(base,   base*2)  # s1: 2b
        self.d2 = Down(base*2, base*4)  # s2: 4b
        self.d3 = Down(base*4, base*8)  # s3: 8b

        # bottleneck (CAN-style dilations)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base*8, base*16, 3, padding=1, dilation=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(base*16, base*16, 3, padding=2, dilation=2, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(base*16, base*16, 3, padding=4, dilation=4, bias=False), nn.ReLU(inplace=True),
        )

        # decoder: upsample to SAME channels as the skip
        self.u3 = Up(base*16, base*4)  # skip s2: 4b
        self.u2 = Up(base*8,  base*2)  # skip s1: 2b
        self.u1 = Up(base*4,  base)    # skip s0: 1b

        # heads
        self.head = nn.Sequential(
            conv3x3(base, base), nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.Conv2d(base, n_classes, 1)
        )

        if self.deep:
            self.aux2 = nn.Conv2d(base*2, n_classes, 1)  # x2 has 2b now
            self.aux1 = nn.Conv2d(base,   n_classes, 1)  # x1 has 1b

    def forward(self, x):
        s0 = self.stem(x)     # b
        s1 = self.d1(s0)      # 2b
        s2 = self.d2(s1)      # 4b
        s3 = self.d3(s2)      # 8b
        b  = self.bottleneck(s3)

        x3 = self.u3(b,  s2)  # -> 4b
        x2 = self.u2(x3, s1)  # -> 2b
        x1 = self.u1(x2, s0)  # -> 1b

        out = self.head(x1)   # expects 1b
        if self.deep and self.training:
            return out, self.aux2(x2), self.aux1(x1)
        return out

##heyyyy