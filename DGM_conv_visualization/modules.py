import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
import numpy
import math
import cv2
from torchvision.transforms import Resize, functional

class Split_freq(nn.Module):
    def __init__(self, channel_num: int, mode="ideal"):
        # mode: Type of filter, you can choose ideal, gaussian or butterworth
        super(Split_freq, self).__init__()
        self.channel_num = channel_num
        self.mode = mode
        self.mask = self.generate_freq_mask(1024, 1024)
    
    def generate_freq_mask(self, H, W):
        length = math.sqrt((H/2)**2+(W/2)**2)
        length_interval = length / self.channel_num
        pf_chunk = []
        if self.mode == "ideal":
            for i in range(self.channel_num):
                pf = numpy.zeros((H, W))
                cv2.circle(pf, (W//2, H//2), math.ceil((i+1)*length_interval), (1), -1)
                pf = torch.from_numpy(pf).float().unsqueeze(0)
                if i == 0:
                    pass
                else:
                    for prev in pf_chunk:
                        pf = pf - prev
                pf_chunk.append(pf)
        elif self.mode == "gaussian":
            a0 = H//2
            b0 = W//2
            for n in range(self.channel_num):
                # pf = numpy.zeros((H, W))
                h_list = numpy.arange(-a0,H-a0,1)**2
                w_list = numpy.arange(-b0,W-b0,1)**2
                pf = numpy.zeros((H, W))
                for i in range(h_list.shape[0]):
                    pf[i, :] = h_list[i] + w_list
                pf = numpy.sqrt(pf)
                pf = numpy.exp(-numpy.power(pf, 2)/(2*((length_interval*(n+1))**2)))
                pf = torch.from_numpy(pf).float().unsqueeze(0)
                if n == 0:
                    pass
                else:
                    for prev in pf_chunk:
                        pf = pf - prev
                pf_chunk.append(pf)
        elif self.mode == "butterworth":
            a0 = H//2
            b0 = W//2
            n = 2
            for n in range(self.channel_num):
                # pf = numpy.zeros((H, W))
                h_list = numpy.arange(-a0,H-a0,1)**2
                w_list = numpy.arange(-b0,W-b0,1)**2
                pf = numpy.zeros((H, W))
                for i in range(h_list.shape[0]):
                    pf[i, :] = h_list[i] + w_list
                pf = numpy.sqrt(pf)
                pf = 1/(1+numpy.power((pf/(length_interval*(n+1))), 2*(n+1)))
                pf = torch.from_numpy(pf).float().unsqueeze(0)
                if n == 0:
                    pass
                else:
                    for prev in pf_chunk:
                        pf = pf - prev
                pf_chunk.append(pf)
        else:
            raise TypeError("Wrong filter mode!")
        pf_chunk = torch.cat(pf_chunk, dim=0)
        return pf_chunk
    
    def forward(self, x):
        B, C, H, W = x.size()
        x_list = torch.split(x, 1, dim=1)
        mask = Resize([H, W], interpolation=functional.InterpolationMode.BICUBIC)(self.mask).to(x.device)
        out_list = []
        for x in x_list:
            f = fft.fftn(x, dim=(2,3))
            f = fft.fftshift(f, dim=(2,3))
            f_split = f * mask
            f_split = fft.ifftshift(f_split, dim=(2,3))
            out = fft.ifftn(f_split, dim=(2,3)).real
            out_list.append(out.unsqueeze(-1))
        out = torch.cat(out_list, dim=-1)
        out = torch.split(out, 1, dim=1)
        outs = []
        for f in out:
            outs.append(f.squeeze(1).permute(0, 3, 1, 2).contiguous())
        # out = torch.sqrt(torch.pow(out.real, 2)+torch.pow(out.imag, 2))
        return outs, mask

class bicubic_imresize(nn.Module):
    """
    An pytorch implementation of imresize function in MATLAB with bicubic kernel.
    """
    def __init__(self, scale_factor):
        super(bicubic_imresize, self).__init__()
        self.scale = scale_factor

    def cubic(self, x):
        absx = torch.abs(x)
        absx2 = torch.abs(x) * torch.abs(x)
        absx3 = torch.abs(x) * torch.abs(x) * torch.abs(x)

        condition1 = (absx <= 1).to(torch.float32)
        condition2 = ((1 < absx) & (absx <= 2)).to(torch.float32)

        f = (1.5 * absx3 - 2.5 * absx2 + 1) * condition1 + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * condition2
        return f

    def contribute(self, in_size, out_size, scale, cuda_flag):
        kernel_width = 4
        if scale < 1:
            kernel_width = 4 / scale
        x0 = torch.arange(start=1, end=out_size[0] + 1).to(torch.float32)
        x1 = torch.arange(start=1, end=out_size[1] + 1).to(torch.float32)
        if cuda_flag:
            x0 = x0.cuda()
            x1 = x1.cuda()

        u0 = x0 / scale + 0.5 * (1 - 1 / scale)
        u1 = x1 / scale + 0.5 * (1 - 1 / scale)

        left0 = torch.floor(u0 - kernel_width / 2)
        left1 = torch.floor(u1 - kernel_width / 2)

        P = numpy.ceil(kernel_width) + 2

        if cuda_flag:
            indice0 = left0.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0).cuda()
            indice1 = left1.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0).cuda()
        else:
            indice0 = left0.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0)
            indice1 = left1.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0)

        mid0 = u0.unsqueeze(1) - indice0.unsqueeze(0)
        mid1 = u1.unsqueeze(1) - indice1.unsqueeze(0)

        if scale < 1:
            weight0 = scale * self.cubic(mid0 * scale)
            weight1 = scale * self.cubic(mid1 * scale)
        else:
            weight0 = self.cubic(mid0)
            weight1 = self.cubic(mid1)

        weight0 = weight0 / (torch.sum(weight0, 2).unsqueeze(2))
        weight1 = weight1 / (torch.sum(weight1, 2).unsqueeze(2))

        if cuda_flag:
            indice0 = torch.min(torch.max(torch.FloatTensor([1]).cuda(), indice0),
                                torch.FloatTensor([in_size[0]]).cuda()).unsqueeze(0)
            indice1 = torch.min(torch.max(torch.FloatTensor([1]).cuda(), indice1),
                                torch.FloatTensor([in_size[1]]).cuda()).unsqueeze(0)
        else:
            indice0 = torch.min(torch.max(torch.FloatTensor([1]), indice0),
                                torch.FloatTensor([in_size[0]])).unsqueeze(0)
            indice1 = torch.min(torch.max(torch.FloatTensor([1]), indice1),
                                torch.FloatTensor([in_size[1]])).unsqueeze(0)

        kill0 = torch.eq(weight0, 0)[0][0]
        kill1 = torch.eq(weight1, 0)[0][0]

        weight0 = weight0[:, :, kill0 == 0]
        weight1 = weight1[:, :, kill1 == 0]

        indice0 = indice0[:, :, kill0 == 0]
        indice1 = indice1[:, :, kill1 == 0]

        return weight0, weight1, indice0, indice1

    def forward(self, input):
        if len(input.shape) == 3:
            input = input.unsqueeze(0)
        [b, c, h, w] = input.shape
        output_size = [b, c, int(h * self.scale), int(w * self.scale)]
        cuda_flag = input.is_cuda

        weight0, weight1, indice0, indice1 = self.contribute([h, w], [int(h * self.scale), int(w * self.scale)], self.scale, cuda_flag)
        weight0 = weight0.squeeze(0)
        indice0 = indice0.squeeze(0).long()
        out = input[:, :, (indice0 - 1), :] * (weight0.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = (torch.sum(out, dim=3))
        A = out.permute(0, 1, 3, 2)

        weight1 = weight1.squeeze(0)

        indice1 = indice1.squeeze(0).long()
        out = A[:, :, (indice1 - 1), :] * (weight1.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = torch.sum(out, dim=3).permute(0, 1, 3, 2)
        return out

def _pos(p, pos_fn):
    pos_fn = pos_fn.lower()
    if pos_fn == 'softmax':
        p_sz = p.size()
        p = p.view(p_sz[0], p_sz[1], -1)
        p = F.softmax(p, -1)
        return p.view(p_sz)
    elif pos_fn == 'exp':
        return torch.exp(p)
    elif pos_fn == 'softplus':
        return F.softplus(p, beta=10)
    elif pos_fn == 'sigmoid':
        return F.sigmoid(p)
    else:
        print('Undefined positive function!')
        return

#### Non-negativity enforcement function for the applicability function
class EnforcePos(object):
    def __init__(self, name, pos_fn):
        self.name = name
        self.pos_fn = pos_fn

    def compute_weight(self, module):
        return _pos(getattr(module, self.name + '_p'), self.pos_fn)

    @staticmethod
    def apply(module, name, pos_fn):
        fn = EnforcePos(name, pos_fn)

        weight = getattr(module, name)

        # remove w from parameter list
        del module._parameters[name]

        #
        module.register_parameter(name + '_p', nn.Parameter(_pos(weight, pos_fn).data))
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_p']
        module.register_parameter(self.name, nn.Parameter(weight.data))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))

##### Normalized Convolution Layer
class NConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=None, dilation=(1, 1), groups=1,
                 bias=False, pos_fn='softplus', prop_conf=True, init_method='n'):
        if padding is None:
            padding = _pair(kernel_size[0] // 2)
        super(NConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, transposed=False,
                                      output_padding=_pair(0), groups=groups, bias=bias, padding_mode='zeros', )
        self.eps = 1e-20
        self.pos_fn = pos_fn
        self.init_method = init_method
        self.prop_conf = prop_conf

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize weights and bias
        self.init_parameters()

        if self.pos_fn is not None:
            EnforcePos.apply(self, 'weight', pos_fn)

    def forward(self, inpt):
        data = inpt[0]
        conf = inpt[1]

        # Normalized Convolution
        denom = F.conv2d(conf, self.weight, None, self.stride,
                         self.padding, self.dilation, self.groups)
        nomin = F.conv2d(data * conf, self.weight, None, self.stride,
                         self.padding, self.dilation, self.groups)
        nconv = nomin / (denom + self.eps)

        # Add bias
        if self.bias is not None:
            b = self.bias.view(1, len(self.bias), 1, 1)
            b = b.expand_as(nconv)
            nconv += b
        if self.prop_conf:
            # Propagate confidence
            cout = denom
            sz = cout.size()
            cout = cout.view(sz[0], sz[1], -1)

            k = self.weight
            k_sz = k.size()
            k = k.view(k_sz[0], -1)
            s = torch.sum(k, dim=-1, keepdim=True)
            #k = k.view(k_sz[0], k_sz[1], -1)
            #s = torch.sum(k, dim=-1, keepdim=True).squeeze(-1)

            cout = cout / s
            cout = cout.view(sz)

        else:
            cout = None

        return nconv, cout

    def init_parameters(self):
        # Init weights
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.weight)
        elif self.init_method == 'k':  # Kaiming
            torch.nn.init.kaiming_uniform_(self.weight)
        elif self.init_method == 'n':  # Normal dist
            n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
            self.weight.data.normal_(2, math.sqrt(2. / n))

        # Init bias
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

class NconvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=None, dilation=(1, 1), groups=1,
                 bias=False, pos_fn='softplus', prop_conf=True, init_method='n', generate_mode="matrix"):
        super(NconvModule, self).__init__()
        '''
        The input of NConv2d must be with images and confidence weights, so we uesd two ways to generate the confidence weights.
        generate_mode (str): The way to generate the confidence weights, you can choose 'net' to use a convolution layer to generate confidence weights, and 'matrix' to use the all 1 matrix as confidence weights.
        '''
        self.mode = generate_mode
        self.nconv = NConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                 bias, pos_fn, prop_conf, init_method)
        if self.mode == "net":
            self.gen_weights = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                                             nn.ReLU(),)
        elif self.mode == "matrix":
            pass
        else:
            raise ValueError("generate_mode must be 'matrix' or 'net'")
    
    def forward(self, x):
        if self.mode == "net":
            w = self.gen_weights(x)
        elif self.mode == "matrix":
            w = torch.ones_like(x).to(x.device)
        x = (x, w)
        out, _ = self.nconv(x)
        return out

class Displacement_generate(nn.Module):
    def __init__(self, scale, mode="bicubic", color_channel=1, conv="default"):
        '''
        Displacement generate module operates similarly to pre-upsampling by first upsampling the LR image to the target resolution. The image is then re-transformed back into the low-dimensional space by constructing an inverse operation of subpixel layer, which also transforms the subpixel information generated during the upsampling into the channel dimension.
        scale (int): Scale factor
        mode (str): The mode of interpolation, you can select "bicubic", "bilinear" and "nearest"
        color_channel (int): Input channels
        conv (str): The conv in DGM, you can choose "default", "common" and "normalized". The "default" is the conv layer in the paper, the "common" is a ordinary convolutional layer, the "normalized-net" is a normalized convolution layer with a convolution layer to generate confidence weights, and "normalized" is a normalized convolution layer with a all 1 matrix as confidence weights.
        '''
        super(Displacement_generate, self).__init__()
        self.scale = scale
        if mode == "bicubic":
            self.up = bicubic_imresize(scale_factor=scale)
        elif mode == "nearest":
            self.up = nn.UpsamplingNearest2d(scale_factor=scale)
        elif mode == "bilinear":
            self.up = nn.UpsamplingBilinear2d(scale_factor=scale)
        else:
            raise ValueError("mode must be 'bicubic', 'bilinear' or 'nearest'")
        
        if conv == "default":
            self.displacement = nn.Conv2d(color_channel, color_channel*scale*scale, scale, scale, 0, groups=color_channel, bias=False)
            p = numpy.arange(scale*scale, dtype="int64")
            p = torch.from_numpy(p)
            p = F.one_hot(p)
            p = p.view(scale*scale, 1, scale, scale).float()
            p = p.repeat(color_channel, 1, 1, 1)
            for m in self.modules():
                if m == self.displacement:
                    m.weight.data = p
            for params in self.parameters():
                params.requires_grad = False
        elif conv == "normalized":
            self.displacement = NconvModule(in_channels=color_channel, out_channels=color_channel*scale*scale, kernel_size=(scale,scale), stride=scale, padding=(0,0), groups=color_channel, bias=False, prop_conf=False, generate_mode="matrix")
        elif conv == "normalized_net":
            self.displacement = NconvModule(in_channels=color_channel, out_channels=color_channel*scale*scale, kernel_size=(scale,scale), stride=scale, padding=(0,0), groups=color_channel, bias=False, prop_conf=False, generate_mode="net")
        elif conv == "common":
            self.displacement = nn.Conv2d(color_channel, color_channel*scale*scale, scale, scale, 0, groups=color_channel, bias=False)
        else:
            raise ValueError("conv must be 'default', 'normalized' or 'common'")
    
    def forward(self, x):
        x = self.up(x)
        x = self.displacement(x)
        return x
