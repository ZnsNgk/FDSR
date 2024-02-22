import torch
import cv2
import numpy
import os
from FDSR_4conv import FDSR

mode_list = ["conv", "nconv_matrix", "nconv_net", "ours"]
mode_list_in_paper = ["Conv", "NConv-C=1", "NConv-conv", "Ours"]
model_info = [{"scale": 3, "freq_c": 8, "c": 64, "mode": "ideal", "color_channel": 3, "DGM_up_method": "bicubic", "conv_in_DGM": "common"},
              {"scale": 3, "freq_c": 8, "c": 64, "mode": "ideal", "color_channel": 3, "DGM_up_method": "bicubic", "conv_in_DGM": "normalized"},
              {"scale": 3, "freq_c": 8, "c": 64, "mode": "ideal", "color_channel": 3, "DGM_up_method": "bicubic", "conv_in_DGM": "normalized_net"},
              {"scale": 3, "freq_c": 8, "c": 64, "mode": "ideal", "color_channel": 3, "DGM_up_method": "bicubic", "conv_in_DGM": "default"}]

# 加载图像
hr_image = cv2.imread("barbara.png")
hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
hr_image = torch.from_numpy(hr_image).permute(2, 0, 1).unsqueeze(0).float() /255.
subpix_layer = torch.nn.PixelShuffle(3)
loss_func = torch.nn.MSELoss()

for mode, mode_in_paper, kwargs in zip(mode_list, mode_list_in_paper, model_info):
    model = FDSR(**kwargs)
    try:
        model.load_state_dict(torch.load(mode+".pth", map_location=torch.device("cpu")))
    except:
        print(mode + "load failed")
        continue
    model.eval()
    forward_func = model.displacement.displacement
    with torch.no_grad():
        subpixel_image = forward_func(hr_image)
        rec_image = subpix_layer(subpixel_image)
    loss = loss_func(rec_image, hr_image)
    # 保存图片
    if not os.path.exists(mode_in_paper):
        os.mkdir(mode_in_paper)
    rec_image = rec_image.squeeze().permute(1, 2, 0).cpu().numpy()*255.
    rec_image = cv2.cvtColor(rec_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(mode_in_paper + "/rec_image.png", rec_image)
    with open(mode_in_paper + "/MSE_loss.txt", "w") as f:
        f.write(str(loss.item()))
    R, G, B = torch.split(subpixel_image.squeeze(), 9, dim=0)
    R = torch.split(R, 1, dim=0)
    G = torch.split(G, 1, dim=0)
    B = torch.split(B, 1, dim=0)
    subpixel_image = []
    for r, g, b in zip(R, G, B):
        rgb_image = torch.cat([r, g, b], dim=0)
        subpixel_image.append(rgb_image)
    subpixel_image = torch.cat(subpixel_image, dim=0)
    subpixel_image_split = torch.split(subpixel_image.squeeze(), 3, dim=0)
    subpixel_image_col_cat = []
    subpixel_image_row_cat = []
    for idx, subpixel_image_s in enumerate(subpixel_image_split):
        subpixel_image_col_cat.append(subpixel_image_s)
        if idx % 3 == 2:
            subpixel_image_row = torch.cat(subpixel_image_col_cat, dim=1)
            subpixel_image_row_cat.append(subpixel_image_row)
            subpixel_image_col_cat = []
    subpixel_image_row_cat = torch.cat(subpixel_image_row_cat, dim=2)
    subpixel_image = subpixel_image_row_cat.permute(1, 2, 0).cpu().numpy()*255
    subpixel_image = cv2.cvtColor(subpixel_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(mode_in_paper + "/pixshift_image.png", subpixel_image)
