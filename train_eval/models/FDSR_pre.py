import torch
import torch.nn as nn
from .modules import Split_freq

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
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.act4(x)
        return x

class FDSR_pre(nn.Module):
    def __init__(self, freq_c=8, c=64, mode="ideal", color_channel=3, freq_order="l2h"):
        super(FDSR_pre, self).__init__()
        self.freq_c = freq_c
        self.c = c
        if freq_order == "h2l":
            self.freq_rev = True
        elif freq_order == "l2h":
            self.freq_rev = False
        else:
            raise ValueError("Frequency Order can only choose 'low to high'(l2h) or 'high to low'(h2l)")
        self.split = Split_freq(freq_c, mode)
        self.rec_blocks = nn.ModuleList()
        for _ in range(freq_c):
            self.rec_blocks.append(Rec_block(out_c=color_channel, in_c=freq_c*color_channel, mid_c=c))
    
    def forward(self, x):
        freq, mask = self.split(x)
        if self.freq_rev:
            freq = freq[::-1]   #frequency order from high to low
        feat_f = []
        for i in range(self.freq_c):
            freq_i = torch.cat(feat_f + list(freq[i:]), dim=1)
            freq_o = self.rec_blocks[i](freq_i)
            feat_f.append(freq_o)
        out_list = []
        for f in feat_f:
            out_list.append(f.unsqueeze(-1))
        out = torch.sum(torch.cat(out_list, dim=-1), dim=-1, keepdim=False)
        return out

if __name__ == "__main__":
    model = FDSR_pre(8, 64)
    x = torch.randn([1, 3, 64, 64])
    y = model(x)
    print(y.size())
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')