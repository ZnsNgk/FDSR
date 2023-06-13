import torch
import math
import numpy
from models.modules import Split_freq

class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self, eps=1e-6):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = eps

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

class L2_Deep_Supervision_Frequency_Division_Loss(torch.nn.Module):
    def __init__(self, freq_c=8, color_channel=3, mode="butterworth"):
        super(L2_Deep_Supervision_Frequency_Division_Loss, self).__init__()
        self.color_channel = color_channel
        self.loss = torch.nn.MSELoss()
        self.Split = Split_freq(freq_c, mode=mode)
        
    def forward(self, X, Y):
        x_up, X_f = X
        Y_f, _ = self.Split(Y)
        # Y_f = torch.split(Y, self.color_channel, dim=1)
        losses = []
        losses.append(self.loss(x_up, Y))
        for x, y in zip(X_f, Y_f):
            l = self.loss(x, y)
            losses.append(l)
        return sum(losses)

class L1_Deep_Supervision_Frequency_Division_Loss(torch.nn.Module):
    def __init__(self, freq_c=8, color_channel=3, mode="butterworth"):
        super(L1_Deep_Supervision_Frequency_Division_Loss, self).__init__()
        self.color_channel = color_channel
        self.loss = torch.nn.L1Loss()
        self.Split = Split_freq(freq_c, mode=mode)
        
    def forward(self, X, Y):
        _, X_f = X
        Y_f, _ = self.Split(Y)
        # Y_f = torch.split(Y, self.color_channel, dim=1)
        losses = []
        for x, y in zip(X_f, Y_f):
            l = self.loss(x, y)
            losses.append(l)
        return sum(losses)

class L1C_Deep_Supervision_Frequency_Division_Loss(torch.nn.Module):
    def __init__(self, freq_c=8, color_channel=3, eps=1e-6, mode="butterworth"):
        super(L1C_Deep_Supervision_Frequency_Division_Loss, self).__init__()
        self.color_channel = color_channel
        self.loss = L1_Charbonnier_loss(eps)
        self.Split = Split_freq(freq_c, mode=mode)
        
    def forward(self, X, Y):
        x_up, X_f = X
        Y_f, _ = self.Split(Y)
        # Y_f = torch.split(Y, self.color_channel, dim=1)
        losses = []
        losses.append(self.loss(x_up, Y))
        for x, y in zip(X_f, Y_f):
            l = self.loss(x, y)
            losses.append(l)
        return sum(losses)

loss_func_list = {
    "L1": torch.nn.L1Loss,
    "L2": torch.nn.MSELoss,
    "L1_Charbonnier": L1_Charbonnier_loss,
    "L2_FDL": L2_Deep_Supervision_Frequency_Division_Loss,
    "L1C_FDL": L1C_Deep_Supervision_Frequency_Division_Loss,
    "L1_FDL": L1_Deep_Supervision_Frequency_Division_Loss
}

def get_loss_func(name, **kwargs):
    return loss_func_list[name](**kwargs)

def loss_PSNR_L1(loss):
    return 10. * math.log10(65025. / (loss**2))

def loss_PSNR_L1_norm(loss):
    return 10. * math.log10(1. / (loss**2))

def loss_PSNR_L2(loss):
    return 10. * math.log10(65025. / loss)

def loss_PSNR_L2_norm(loss):
    return 10. * math.log10(1. / loss)

cal_list = {
    "L1": loss_PSNR_L1,
    "L2": loss_PSNR_L2,
    "L1_Charbonnier": loss_PSNR_L1,
    "L2_FDL": loss_PSNR_L2,
    "L1C_FDL": loss_PSNR_L1,
    "L1_FDL": loss_PSNR_L1,
}

cal_list_norm = {
    "L1": loss_PSNR_L1_norm,
    "L2": loss_PSNR_L2_norm,
    "L1_Charbonnier": loss_PSNR_L1_norm,
    "L2_FDL": loss_PSNR_L2_norm,
    "L1C_FDL": loss_PSNR_L1_norm,
    "L1_FDL": loss_PSNR_L1_norm,
}

def get_loss_PSNR(name, normal):
    if normal:
        return cal_list_norm[name]
    else:
        return cal_list[name]