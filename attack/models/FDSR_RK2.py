import torch
import torch.nn as nn
from .modules import Displacement_generate, Split_freq

class Rec_Block(nn.Module):
    def __init__(
        self, in_c, out_c, n_feats, kernel_size, bias=True):
        super(Rec_Block, self).__init__()
        self.conv_in = nn.Conv2d(in_c, n_feats, 1, 1, 0, bias=True)
        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2), bias=bias)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2), bias=bias)
        self.conv3 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2), bias=bias)
        self.conv4 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2), bias=bias)
        self.conv_out = nn.Conv2d(n_feats, out_c, 1, 1, 0, bias=True)
        self.relu1 = nn.PReLU(n_feats, 0.25)
        self.relu2 = nn.PReLU(n_feats, 0.25)
        self.relu3 = nn.PReLU(n_feats, 0.25)
        self.relu4 = nn.PReLU(n_feats, 0.25)
        self.scale1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)

    def forward(self, x):
        x = self.conv_in(x)
        yn = x
        G_yn = self.relu1(x)
        G_yn = self.conv1(G_yn)
        G_yn = self.relu3(G_yn)
        G_yn = self.conv3(G_yn)
        yn_1 = G_yn + yn
        Gyn_1 = self.relu2(yn_1)
        Gyn_1 = self.conv2(Gyn_1)
        Gyn_1 = self.relu4(Gyn_1)
        Gyn_1 = self.conv4(Gyn_1)
        yn_2 = Gyn_1 + G_yn
        yn_2 = yn_2*self.scale1
        out = yn_2 + yn
        out = self.conv_out(out)
        return out

class FDSR(nn.Module):
    def __init__(self, scale, freq_c=8, c=64, mode="ideal", color_channel=3):
        super(FDSR, self).__init__()
        self.color_channel = color_channel
        self.scale = scale
        self.freq_c = freq_c
        self.c = c
        self.displacement = Displacement_generate(scale, "bicubic", color_channel=color_channel)
        self.split = Split_freq(freq_c, mode)
        self.rec_blocks = nn.ModuleList()
        for _ in range(freq_c):
            self.rec_blocks.append(Rec_Block(out_c=color_channel*scale*scale, in_c=freq_c*color_channel*scale*scale, n_feats=c, kernel_size=3))
        if (scale == 2) or (scale == 3) or (scale == 4):
            self.upsample = nn.PixelShuffle(scale)
        else:
            raise NotImplementedError
    
    def forward(self, x):
        x = self.displacement(x)
        freq, mask = self.split(x)
        # mask_n = torch.split(mask, 1, dim=0)
        # freq_n = torch.split(freq, self.color_channel*self.scale*self.scale, dim=1)
        feat_f = []
        for i in range(self.freq_c):
            freq_i = torch.cat(feat_f + list(freq[i:]), dim=1)
            freq_o = self.rec_blocks[i](freq_i)
            feat_f.append(freq_o)
        out_list = []
        for f in feat_f:
            out_list.append(f.unsqueeze(-1))
        out = torch.sum(torch.cat(out_list, dim=-1), dim=-1, keepdim=False)
        out = self.upsample(out)
        # print(out_list[1].size())
        return out

if __name__ == "__main__":
    model = FDSR(4, 32, 64, color_channel=3)
    x = torch.randn([1, 3, 64, 64])
    y = model(x)
    print(y.size())
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    # m = Displacement_generate(3, "bicubic", 3)
    # import cv2
    # img = cv2.imread("./data/test/Set14_LR/3/barbara.png")
    # img = torch.from_numpy(img).float()
    # img = img.permute(2, 0, 1).unsqueeze(0)
    # dis = m(img)
    # print(dis.size())
    # dis = dis.squeeze(0).permute(1, 2, 0)
    # dis = torch.split(dis, 9, dim=2)
    # print(len(dis))
    # R = []
    # G = []
    # B = []
    # for i in range(9):
    #     R.append(dis[0][:, :, i])
    #     G.append(dis[1][:, :, i])
    #     B.append(dis[2][:, :, i])
    # num = 1
    # for r, g, b in zip(R, G, B):
    #     i = torch.cat([r.unsqueeze(-1), g.unsqueeze(-1), b.unsqueeze(-1)], dim=2)
    #     cv2.imwrite("./dis_"+str(num)+".png", i.numpy())
    #     num += 1
