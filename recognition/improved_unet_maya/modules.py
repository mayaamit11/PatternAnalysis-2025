import torch
import torch.nn as nn

def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)
    )

#test comment *
class UNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.d1 = conv_block(in_ch, base);   self.p1 = nn.MaxPool2d(2)
        self.d2 = conv_block(base, base*2);  self.p2 = nn.MaxPool2d(2)
        self.d3 = conv_block(base*2, base*4);self.p3 = nn.MaxPool2d(2)
        self.bn = conv_block(base*4, base*8)
        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, 2); self.c3 = conv_block(base*8, base*4)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, 2); self.c2 = conv_block(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base,   2, 2); self.c1 = conv_block(base*2, base)
        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))
        bn = self.bn(self.p3(d3))
        x = self.c3(torch.cat([self.u3(bn), d3], 1))
        x = self.c2(torch.cat([self.u2(x),  d2], 1))
        x = self.c1(torch.cat([self.u1(x),  d1], 1))
        return self.head(x)  # logits
