import torch
import torch.nn as nn

class Rec_block(nn.Module):
    def __init__(self, out_c, in_c, mid_c, res=False):
        super(Rec_block, self).__init__()
        self.res = res
        self.conv1 = nn.Conv2d(in_c, mid_c, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(mid_c, mid_c, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(mid_c, out_c, 3, 1, 1, bias=True)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.act3 = nn.PReLU()
        # self.act4 = nn.PReLU()
        # self.att = Att_block()
    
    def forward(self, x):
        x_ori = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.conv4(x)
        # x = self.act4(x)
        # x = self.att(x, mask)
        if self.res:
            return x + x_ori
        else:
            return x

class FDSR_wStep(nn.Module):
    def __init__(self, scale, c=64, freq_c=8, color_channel=3, freq_order="l2h"):
        super(FDSR_wStep, self).__init__()
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
        self.split = nn.Conv2d(color_channel, freq_c*color_channel, 1, 1, 0, groups=color_channel, bias=False)
        self.rec_blocks = nn.ModuleList()
        if (scale == 2) or (scale == 3):
            self.upsample = nn.Sequential(
                nn.Conv2d(color_channel, color_channel*scale*scale, 1, 1, 0, bias=False),
                nn.PixelShuffle(scale)
            )
        elif scale == 4:
            self.upsample =  nn.Sequential(
                nn.Conv2d(color_channel, color_channel*4, 1, 1, 0, bias=False),
                nn.PixelShuffle(2),
                nn.Conv2d(color_channel, color_channel*4, 1, 1, 0, bias=False),
                nn.PixelShuffle(2),
            )
        else:
            raise NotImplementedError
        for _ in range(freq_c):
            self.rec_blocks.append(Rec_block(out_c=color_channel, in_c=freq_c*color_channel, mid_c=c))
    
    def forward(self, x):
        freq = self.split(x)
        freq_n = torch.split(freq, self.color_channel, dim=1)
        if self.freq_rev:
            freq_n = freq_n[::-1]   #frequency order from high to low
        feat_f = []
        for i in range(self.freq_c):
            freq_i = torch.cat(feat_f + list(freq_n[i:]), dim=1)
            freq_o = self.rec_blocks[i](freq_i)
            feat_f.append(freq_o)
        out_list = []
        for f in feat_f:
            out_list.append(f.unsqueeze(-1))
        out = torch.sum(torch.cat(out_list, dim=-1), dim=-1, keepdim=False)
        out_tensor = self.upsample(out)
        return out_tensor

