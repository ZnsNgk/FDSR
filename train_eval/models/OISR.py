import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.PReLU(1, 0.25), res_scale=1):

        super(ResBlock, self).__init__()

        self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv3 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv4 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.relu1 = nn.PReLU(n_feats, 0.25)
        self.relu2 = nn.PReLU(n_feats, 0.25)
        self.relu3 = nn.PReLU(n_feats, 0.25)
        self.relu4 = nn.PReLU(n_feats, 0.25)
        self.scale1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)

    def forward(self, x):
        
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
        return out

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class OISR(nn.Module):
    def __init__(self, scale, n_resblocks=8, rgb_range=1., n_feats=122, n_colors=3, conv=default_conv):
        super(OISR, self).__init__()

        n_resblocks = n_resblocks
        n_feats = n_feats
        kernel_size = 3 
        scale = scale
        act = nn.PReLU()
        # self.url = url['r{}f{}x{}'.format(n_resblocks, n_feats, scale)]
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=1
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

if __name__ == "__main__":
    scale = 2
    model = OISR(scale=scale)
    x = torch.randn([1, 1, 3, 480//scale, 360//scale])
    from thop import profile, clever_format
    with torch.no_grad():
        flops, params = profile(model, inputs=x)
        flops, params = clever_format([flops, params], "%.6f")
    print(flops)
    print(params)
