# FDSR_PyTorch

![Stars](https://img.shields.io/github/stars/ZnsNgk/FDSR)
[![Visits Badge](https://badges.pufler.dev/visits/ZnsNgk/FDSR)](https://badges.pufler.dev/visits/ZnsNgk/FDSR)
![GitHub forks](https://img.shields.io/github/forks/ZnsNgk/FDSR?color=blue&label=Forks) 

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
python test.py FDSR_HesRFA_32_h2l --once net_x2_1000.pth
python test.py FDSR_HesRFA_32_h2l --once net_x3_1000.pth
python test.py FDSR_HesRFA_32_h2l --once net_x4_1000.pth

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
python demo.py FDSR_HesRFA_32_h2l --file net_x4_1000.pth --dataset Manga109
python demo.py FDSR_HesRFA_32_FDL_l2_gaus --file net_x4_1000.pth --dataset Manga109

# Self-selected image reconstruction
python demo.py FDSR_HesRFA_32_h2l --file net_x4_1000.pth --input
python demo.py FDSR_HesRFA_32_FDL_l2_gaus --file net_x4_1000.pth --input
```

`--file`  is the file that holds the model weights in the format pth or pkl. `--dataset` is the dataset you want to rebuild.

If you want to rebuild your own image, copy the image to the `demo_input` folder and use `--input`.

The reconstructed image will be saved in the `demo_output` folder.

### Train

You can retrain FDSR and FDSR w/ FDL with the following commands.

```
# Train FDSR
python train.py FDSR_HesRFA_32_h2l

# Train FDSR w/ FDL
python train.py FDSR_HesRFA_32_FDL_l2_gaus
```

If you have multiple GPUs, you should first modify the JSON file in the `config` folder to change the `parallel_mode` there from `DP` to `DDP`, and then start the training with the following commands:

```
# Train FDSR
python -m torch.distributed.launch --nproc_per_node=2 train.py FDSR_HesRFA_32_h2l

# Train FDSR w/ FDL
python -m torch.distributed.launch --nproc_per_node=2 train.py FDSR_HesRFA_32_FDL_l2_gaus
```

## Visualisation

The visualisation section is placed in the 'visualisation' folder and the code can be run as follows to obtain a visualisation of the 'comic' for FDSR and FDSR wFDL at x2 upscale factor.

```
cd visualisation
python main.py
```

## Migration adversarial attacks

The papers and code for the adversarial attack section are taken from <a href="https://github.com/idearibosome/tf-sr-attack">(code)</a> and:

```
@inproceedings{choi2019evaluating,
  title={Evaluating robustness of deep image super-resolution against adversarial attacks},
  author={Choi, Jun-Ho and Zhang, Huan and Kim, Jun-Hyuk and Hsieh, Cho-Jui and Lee, Jong-Seok},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={303--311},
  year={2019}
}
```

If you need to run this part of the code, you will first need to install the following packages:

```
tensorflow 1.12+ (<2.0)
```

And open the following folders:

```
cd attack
```

First, you need to generate a sample of the attack, and you need to execute the following commands in sequence:

```
cd tf-sr-attack
./run_all.sh
```

(If your operating system is Windows, you will need to run `run_all.bat`)

After the adversarial sample has been generated, you will need to fall back to the parent directory and run the following commands in sequence to test:

```
cd ..
python check_folder.py
./run_test.sh
```

(If your operating system is Windows, you will need to run `run_test.bat`)

Finally, you can run `drew_pic.py` to generate the results of the experiment.

## DGM_conv_visualization

This is a visualization of the experiments in subsection 4.4.3 of the paper for the use of different convolutional layers in the DGM. We trained and visualized the model using two different normalized convolutions, ordinary convolution and our proposed convolution, respectively. In this case, all `.pth` files are our trained models, and the DGM convolutional layer in the above models can be visualized by simply running the following code:

```
python main.py
```

