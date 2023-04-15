# FDSR_PyTorch

This repository is an official PyTorch implementation of the paper "FDSR: An Interpretable Frequency Division Network for Single Image Super Resolution".

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

The paper is still under review and only the code for visualisation and migration adversarial attacks has been given so far, the full code will be released after the paper is accepted. Train and test dataset can be downloaded by <a href="https://pan.baidu.com/s/1Xw9w3dXDP1QcIRh8j2zM8w?pwd=sike">here</a>.

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

