import torch
import torch.nn as nn
from .modules import Displacement_generate, Split_freq
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, bn=False, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

class DAC(nn.Module):
    def __init__(self, n_channels):
        super(DAC, self).__init__()

        self.mean = nn.Sequential(
            nn.Conv2d(n_channels, n_channels // 16, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 16, n_channels, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
        )
        self.std = nn.Sequential(
            nn.Conv2d(n_channels, n_channels // 16, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 16, n_channels, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
        )

    def forward(self, observed_feat, referred_feat):
        assert (observed_feat.size()[:2] == referred_feat.size()[:2])
        size = observed_feat.size()
        referred_mean, referred_std = calc_mean_std(referred_feat)
        observed_mean, observed_std = calc_mean_std(observed_feat)

        normalized_feat = (observed_feat - observed_mean.expand(
            size)) / observed_std.expand(size)
        referred_mean = self.mean(referred_mean)
        referred_std = self.std(referred_std)
        output = normalized_feat * referred_std.expand(size) + referred_mean.expand(size)
        return output


class MSHF(nn.Module):
    def __init__(self, n_channels, kernel=3):
        super(MSHF, self).__init__()

        pad = int((kernel - 1) / 2)

        self.grad_xx = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=pad,
                                 dilation=pad, groups=n_channels, bias=True)
        self.grad_yy = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=pad,
                                 dilation=pad, groups=n_channels, bias=True)
        self.grad_xy = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=pad,
                                 dilation=pad, groups=n_channels, bias=True)

        for m in self.modules():
            if m == self.grad_xx:
                m.weight.data.zero_()
                m.weight.data[:, :, 1, 0] = 1
                m.weight.data[:, :, 1, 1] = -2
                m.weight.data[:, :, 1, -1] = 1
            elif m == self.grad_yy:
                m.weight.data.zero_()
                m.weight.data[:, :, 0, 1] = 1
                m.weight.data[:, :, 1, 1] = -2
                m.weight.data[:, :, -1, 1] = 1
            elif m == self.grad_xy:
                m.weight.data.zero_()
                m.weight.data[:, :, 0, 0] = 1
                m.weight.data[:, :, 0, -1] = -1
                m.weight.data[:, :, -1, 0] = -1
                m.weight.data[:, :, -1, -1] = 1

        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False

    def forward(self, x):
        fxx = self.grad_xx(x)
        fyy = self.grad_yy(x)
        fxy = self.grad_xy(x)
        hessian = ((fxx + fyy) + ((fxx - fyy) ** 2 + 4 * (fxy ** 2)) ** 0.5) / 2
        return hessian


class rfa_block(nn.Module):
    def __init__(self, n_channels, kernel, bias=False, activation=nn.ReLU(inplace=True)):
        super(rfa_block, self).__init__()

        self.B1 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel, padding=1, bias=bias),
            activation,
        )
        self.B2 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel, padding=1, bias=bias),
            activation,
        )
        self.B3 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel, padding=1, bias=bias),
            activation,
        )
        self.B4 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel, padding=1, bias=bias),
            activation,
        )
        self.out = nn.Conv2d(n_channels * 4, n_channels, 1, 1, 0, bias=True)
    
    def forward(self, x):
        res = x
        B1_out = self.B1(x)
        x_1 = x + B1_out
        B2_out = self.B2(x_1)
        x_2 = x_1 + B2_out
        B3_out = self.B3(x_2)
        x_3 = x_2 + B3_out
        x_4 = self.B4(x_3)
        x_cat = torch.cat([B1_out, B2_out, B3_out, x_4], 1)
        x_out = self.out(x_cat)
        out = res + x_out
        return out

class DiEnDec(nn.Module):
    def __init__(self, n_channels, act=nn.ReLU(inplace=True)):
        super(DiEnDec, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, n_channels * 2, kernel_size=3, padding=1, dilation=1, bias=True),
            act,
            nn.Conv2d(n_channels * 2, n_channels * 4, kernel_size=3, padding=2, dilation=2, bias=True),
            act,
            nn.Conv2d(n_channels * 4, n_channels * 8, kernel_size=3, padding=4, dilation=4, bias=True),
            act,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(n_channels * 8, n_channels * 4, kernel_size=3, padding=4, dilation=4, bias=True),
            act,
            nn.ConvTranspose2d(n_channels * 4, n_channels * 2, kernel_size=3, padding=2, dilation=2, bias=True),
            act,
            nn.ConvTranspose2d(n_channels * 2, n_channels, kernel_size=3, padding=1, dilation=1, bias=True),
            act,
        )
        self.gate = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1)

    def forward(self, x):
        output = self.gate(self.decoder(self.encoder(x)))
        return output


class SingleModule(nn.Module):
    def __init__(self, n_channels, act, attention, out_c, in_c):
        super(SingleModule, self).__init__()
        res_blocks = rfa_block(n_channels=n_channels, kernel=3, activation=act)
        self.in_conv = nn.Conv2d(in_c, n_channels, 1, 1, 0, bias=True)
        self.body_block = res_blocks
        self.attention = attention
        self.out_conv = nn.Conv2d(n_channels, out_c, 1, 1, 0, bias=True)
        if attention:
            self.coder = nn.Sequential(DiEnDec(3, act))
            self.dac = nn.Sequential(DAC(n_channels))
            self.hessian3 = nn.Sequential(MSHF(n_channels, kernel=3))
            self.hessian5 = nn.Sequential(MSHF(n_channels, kernel=5))
            self.hessian7 = nn.Sequential(MSHF(n_channels, kernel=7))
        else:
            self.cbam = CBAM(n_channels)

    def forward(self, x):
        x = self.in_conv(x)
        sz = x.size()
        resin = self.body_block(x)

        if self.attention:
            hessian3 = self.hessian3(resin)
            hessian5 = self.hessian5(resin)
            hessian7 = self.hessian7(resin)
            hessian = torch.cat((torch.mean(hessian3, dim=1, keepdim=True),
                                 torch.mean(hessian5, dim=1, keepdim=True),
                                 torch.mean(hessian7, dim=1, keepdim=True))
                                , 1)
            hessian = self.coder(hessian)
            attention = torch.sigmoid(self.dac[0](hessian.expand(sz), x))
            resout = resin * attention
        else:
            resout = self.cbam(resin)

        output = resout + x
        
        return self.out_conv(output)

class FDSR(nn.Module):
    def __init__(self, scale, freq_c=8, c=64, mode="ideal", color_channel=3, use_FDL=False, DGM_up_method="bicubic", freq_order="l2h"):
        super(FDSR, self).__init__()
        self.color_channel = color_channel
        self.scale = scale
        self.freq_c = freq_c
        self.c = c
        self.use_FDL = use_FDL
        if freq_order == "h2l":
            self.freq_rev = True
        elif freq_order == "l2h":
            self.freq_rev = False
        else:
            raise ValueError("Frequency Order can only choose 'low to high'(l2h) or 'high to low'(h2l)")
        self.displacement = Displacement_generate(scale, DGM_up_method, color_channel=color_channel)
        self.split = Split_freq(freq_c, mode)
        self.rec_blocks = nn.ModuleList()
        for i in range(freq_c):
            if not self.freq_rev:
                self.rec_blocks.append(SingleModule(out_c=color_channel*scale*scale, in_c=freq_c*color_channel*scale*scale, n_channels=64, act=nn.ReLU(), attention=False if i < freq_c//2 else True))
            else:
                self.rec_blocks.append(SingleModule(out_c=color_channel*scale*scale, in_c=freq_c*color_channel*scale*scale, n_channels=64, act=nn.ReLU(), attention=True if i < freq_c//2 else False))
        if (scale == 2) or (scale == 3) or (scale == 4):
            self.upsample = nn.PixelShuffle(scale)
        else:
            raise NotImplementedError
    
    def forward(self, x):
        x = self.displacement(x)
        freq, mask = self.split(x)
        if self.freq_rev:
            freq = freq[::-1]   #frequency order from high to low
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
        if self.use_FDL:
            out_up_list = []
            for f in feat_f:
                out_up_list.append(self.upsample(f))
            return out, out_up_list
        return out

if __name__ == "__main__":
    scale = 2
    model = FDSR(scale=scale ,freq_c=32, c=64, mode="ideal", color_channel=3, freq_order="h2l")
    x = torch.randn([1, 3, 480//scale, 360//scale])
    from thop import profile, clever_format
    with torch.no_grad():
        flops, params = profile(model, inputs=x)
        flops, params = clever_format([flops, params], "%.6f")
    print(flops)
    print(params)

    