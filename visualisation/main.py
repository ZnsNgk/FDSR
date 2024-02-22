import torch
import cv2
import os
import torch.fft as fft
import numpy
import matplotlib.pyplot as plt
from FDSR_HesRFA import FDSR

def main(model_name):
    if model_name == "FDSR_FDL":
    # q = 32
        para_file = torch.load("x2_wFDL_h2l.pth", map_location=torch.device("cpu"))
    elif model_name == "FDSR":
        para_file = torch.load("x2_h2l.pth", map_location=torch.device("cpu"))
    else:
        raise NameError("Unknown model name")
    model = FDSR(2, freq_c=32, c=64, mode="ideal", use_FDL=True, freq_order="h2l")
    model.load_state_dict(para_file)

    img = cv2.cvtColor(cv2.imread("comic.png"), cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.
    print(img.shape)
    with torch.no_grad():
        sr, feat = model(img)
    i = 1
    sr = sr.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()*255.
    sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
    if model_name == "FDSR_FDL":
        if not os.path.exists("./w_FDL"):
            os.makedirs("./w_FDL")
        cv2.imwrite("./w_FDL/sr_img.png", sr)
    elif model_name == "FDSR":
        if not os.path.exists("./wo_FDL"):
            os.makedirs("./wo_FDL")
        cv2.imwrite("./wo_FDL/sr_img.png", sr)
    for f in feat:
        upsample = f.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        upsample = ((upsample-numpy.min(upsample))/(numpy.max(upsample)-numpy.min(upsample)))*255.
        # upsample = upsample * 255.
        if model_name == "FDSR_FDL":
            if not os.path.exists("./w_FDL/feats"):
                os.makedirs("./w_FDL/feats")
            cv2.imwrite("./w_FDL/feats/"+str(i)+".png", cv2.cvtColor(upsample, cv2.COLOR_RGB2BGR))
            i += 1
        elif model_name == "FDSR":
            if not os.path.exists("./wo_FDL/feats"):
                os.makedirs("./wo_FDL/feats")
            cv2.imwrite("./wo_FDL/feats/"+str(i)+".png", cv2.cvtColor(upsample, cv2.COLOR_RGB2BGR))
            i += 1

    # 查看频率域
    i = 1
    for f in feat:
        x_list = torch.split(f, 1, dim=1)
        out_list = []
        for x in x_list:
            f = fft.fftn(x, dim=(2,3))
        
            # f = torch.split(f.unsqueeze(-1), f.shape[2]//2, dim=2)
            # f = torch.cat(f, dim=-1)
            # f = torch.split(f, f.shape[3]//2, dim=3)
            # f = torch.cat(f, dim=-1)
            # f = F.pixel_shuffle(f.permute(0, 4, 2, 3, 1).squeeze(-1), 2)
            
            f = fft.fftshift(f, dim=(2,3))
            out_list.append(f)
        out = torch.cat(out_list, dim=1)
        # out = torch.sum(out, dim=1, keepdim=False).squeeze(0).detach().numpy()
        out = out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        # print(out.shape)
        out = (20*numpy.log10(1. + out))
        out = ((out-numpy.min(out))/(numpy.max(out)-numpy.min(out)))*255.
        out = numpy.array(out, dtype="uint8")
        if model_name == "FDSR_FDL":
            if not os.path.exists("./w_FDL/freq"):
                os.makedirs("./w_FDL/freq")
            cv2.imwrite("./w_FDL/freq/"+str(i)+".png", out)
            i += 1
        elif model_name == "FDSR":
            if not os.path.exists("./wo_FDL/freq"):
                os.makedirs("./wo_FDL/freq")
            cv2.imwrite("./wo_FDL/freq/"+str(i)+".png", out)
            i += 1

def generate_fig():
    folder_list = ["w_FDL", "wo_FDL"]
    image_list = ["feats", "freq"]
    for folder in folder_list:
        title = "FDSR " + folder
        plt.figure(figsize=(256,8))
        plt.suptitle(title)
        for img in image_list:
            if img == "feats":
                n = 0
            elif img == "freq":
                n = 1
            for i in range(1, 33, 1):
                plt.subplot(2, 32, 32*n+i)
                im = cv2.imread(os.path.join(folder, img, str(i)+".png"))
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                plt.imshow(im)
                plt.title(str(i))
                plt.axis('off')
        plt.show()
    
        
if __name__ == "__main__":
    main("FDSR")
    main("FDSR_FDL")
    generate_fig()