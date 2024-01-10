# FDSR_PyTorch

This repository is an official PyTorch implementation of the paper "FDSR: An Interpretable Frequency Division Stepwise Process Based Single-Image Super-Resolution Network".

## Prerequisites:

```
python==3.8
lpips==0.1.4
matplotlib==3.5.3
numpy==1.23.2
opencv-python==4.6.0.66
pandas==1.4.3
Pillow==9.2.0
scipy==1.9.0
torch==1.8.2
torchaudio==0.8.2
torchvision==0.9.2
tqdm
```

## Document

The paper is still under review and only the code for train and test  has been given so far, the full code will be released after the paper is accepted. Train and test dataset can be downloaded by <a href="https://pan.baidu.com/s/1Xw9w3dXDP1QcIRh8j2zM8w?pwd=sike">here</a>.

## Train & Test

### Test

We provide pre-trained FDSR and FDSR w/ FDL, which you can test with the following commands:

```
cd train_eval

# First run the following file to check if the folder is complete
python check_folder.py

# Test FDSR at x2, x3, and x4 upscale factors
python test.py FDSR_HesRFA_32 --once net_x2_1000.pth
python test.py FDSR_HesRFA_32 --once net_x3_1000.pth
python test.py FDSR_HesRFA_32 --once net_x4_1000.pth

# Test FDSR w/ FDL at x2, x3, and x4 upscale factors
python test.py FDSR_HesRFA_32_FDL_l2_gaus --once net_x2_1000.pth
python test.py FDSR_HesRFA_32_FDL_l2_gaus --once net_x3_1000.pth
python test.py FDSR_HesRFA_32_FDL_l2_gaus --once net_x4_1000.pth
```

### Demo

If you want to obtain the output image of FDSR or FDSR w/ FDL, you can do so with the following commands:

```
cd train_eval

# Rebuild a dataset
python demo.py FDSR_HesRFA_32 --file net_x4_1000.pth --dataset Manga109
python demo.py FDSR_HesRFA_32_FDL_l2_gaus --file net_x4_1000.pth --dataset Manga109

# Self-selected image reconstruction
python demo.py FDSR_HesRFA_32 --file net_x4_1000.pth --input
python demo.py FDSR_HesRFA_32_FDL_l2_gaus --file net_x4_1000.pth --input
```

`--file`  is the file that holds the model weights in the format pth or pkl. `--dataset` is the dataset you want to rebuild.

If you want to rebuild your own image, copy the image to the `demo_input` folder and use `--input`.

The reconstructed image will be saved in the `demo_output` folder.

### Train

You can retrain FDSR and FDSR w/ FDL with the following commands.

```
# Train FDSR
python train.py FDSR_HesRFA_32

# Train FDSR w/ FDL
python train.py FDSR_HesRFA_32_FDL_l2_gaus
```

If you have multiple GPUs, you should first modify the JSON file in the `config` folder to change the `parallel_mode` there from `DP` to `DDP`, and then start the training with the following commands:

```
# Train FDSR
python -m torch.distributed.launch --nproc_per_node=2 train.py FDSR_HesRFA_32

# Train FDSR w/ FDL
python -m torch.distributed.launch --nproc_per_node=2 train.py FDSR_HesRFA_32_FDL_l2_gaus
```

## Visualisation

coming soon.

## Migration adversarial attacks

coming soon.

