import torch
import torch.nn as nn
from .modules import Displacement_generate, Split_freq

class Rec_block(nn.Module):
    def __init__(self, out_c, in_c, mid_c):
        super(Rec_block, self).__init__()
        self.conv1 = nn.Conv2d(in_c, mid_c, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(mid_c, mid_c, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(mid_c, out_c, 3, 1, 1, bias=True)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.act3 = nn.PReLU()
        self.act4 = nn.PReLU()
    
    def forward(self, x, mask):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.act4(x)
        return x

class FDSR(nn.Module):
    def __init__(self, scale, freq_c=8, c=64, mode="ideal", color_channel=3, freq_order="l2h", DGM_up_method="bicubic"):
        super(FDSR, self).__init__()
        self.color_channel = color_channel
        self.scale = scale
        self.freq_c = freq_c
        self.c = c
        if freq_order == "h2l":
            self.freq_rev = True
        elif freq_order == "l2h":
            self.freq_rev = False
        else:
            raise ValueError("Frequency Order can only choose 'low to high'(l2h) or 'high to low'(h2l)")
        self.displacement = Displacement_generate(scale, DGM_up_method, color_channel=color_channel)
        self.split = Split_freq(freq_c, mode)
        self.rec_blocks = nn.ModuleList()
        for _ in range(freq_c):
            self.rec_blocks.append(Rec_block(out_c=color_channel*scale*scale, in_c=freq_c*color_channel*scale*scale, mid_c=c))
        if (scale == 2) or (scale == 3) or (scale == 4):
            self.upsample = nn.PixelShuffle(scale)
        else:
            raise NotImplementedError
    
    def forward(self, x):
        x = self.displacement(x)
        freq, mask = self.split(x)
        if self.freq_rev:
            freq = freq[::-1]   #frequency order from high to low
        mask_n = torch.split(mask, 1, dim=0)
        # freq_n = torch.split(freq, self.color_channel*self.scale*self.scale, dim=1)
        feat_f = []
        for i in range(self.freq_c):
            freq_i = torch.cat(feat_f + list(freq[i:]), dim=1)
            freq_o = self.rec_blocks[i](freq_i, mask_n[i])
            feat_f.append(freq_o)
        out_list = []
        for f in feat_f:
            out_list.append(f.unsqueeze(-1))
        out = torch.sum(torch.cat(out_list, dim=-1), dim=-1, keepdim=False)
        out = self.upsample(out)
        # print(out_list[1].size())
        return out
