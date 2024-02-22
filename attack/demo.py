import os
import models
import utils
import cv2
import torch
import json
import numpy
import argparse
import torch.fft as fft

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("cfg_file", help = "config file", type = str)
    parser.add_argument("--file", default = None, type = str, help="Your model parameter file path")
    parser.add_argument("--input", action = "store_true")
    parser.add_argument("--dataset", default = None, type = str)
    parser.add_argument("--show_freq", action= "store_true")
    args = parser.parse_args()
    return args

def load_json(args):
    cfg_file = os.path.join("./config/", args + ".json")
    with open(cfg_file, "r") as f:
        config = json.load(f)
    return config

class Demo():
    def __init__(self, args, config):
        self.model = config["system"]["model_name"]
        self.device = config["system"]["device"]
        self.model_name = args.cfg_file
        self.model_mode = config["system"]["model_mode"]
        self.scale_pos = config["system"]["scale_position"]
        self.is_normal = utils.get_bool(config["dataloader"]["normalize"])
        self.color_seq = "RGB"
        self.show_freq = args.show_freq
        self.is_Y = False
        if config["system"]["color_channel"] == "RGB":
            self.is_Y = False
        elif config["system"]["color_channel"] == "Y":
            self.is_Y = True
        if "color_seq" in config["dataloader"]["data_opts"]:
            self.color_seq = config["dataloader"]["data_opts"]["color_seq"]
        self.model_args = None
        self.file = args.file
        self.file_path = './trained_model/' + self.model_name + '/' + self.file
        self.save_path = './demo_output/' + self.model_name + '/'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.__check_cuda()
        if "model_args" in config["system"]:
            self.model_args = config["system"]["model_args"]
        if "pkl" in self.file:
            self.is_pkl = True
            self.scale = int(self.file.replace(".pkl", "").replace("x", ""))
        else:
            self.is_pkl = False
            if "best" in self.file:
                state_list = self.file.split('_')
                self.scale = int(state_list[0].replace('x', ''))
            else:
                state_list = self.file.split('_')
                self.scale = int(state_list[1].replace('x', ''))
        if args.input:
            self.is_input = True
            self.dataset = './demo_input'
            self.data_name = "input"
        else:
            self.is_input = False
            self.dataset = './data/test/' + args.dataset
            self.data_name = args.dataset
    def __check_cuda(self):
        if "cuda" in self.device:
            if not torch.cuda.is_available():
                print("Cuda is not useable, now try to use cpu!")
                self.device = "cpu"
        self.device = torch.device(self.device)
    def __get_model(self):
        if self.model_mode == "post":
            if self.scale_pos == "init":
                if self.model_args == None:
                    m = models.get_model(self.model, scale=self.scale)
                else:
                    m =  models.get_model(self.model, scale=self.scale, **self.model_args)
                if self.show_freq and "FDSR" in m.__class__.__name__:
                    m.use_FDL = True
                return m
            elif self.scale_pos == "forward":
                if self.model_args == None:
                    return models.get_model(self.model)
                else:
                    return models.get_model(self.model, **self.model_args)
        elif self.model_mode == "pre":
            if self.model_args == None:
                return models.get_model(self.model)
            else:
                return models.get_model(self.model, **self.model_args)
        else:
            raise NameError("ERROR model_mode!")
    def __scale_pos_is_forward(self, net, loader, is_normal):
        for _, data in enumerate(loader):
            img, name = data
            sr = net(img.to(self.device), self.scale)
            sr = sr.permute(0, 2, 3, 1).squeeze(0)
            if is_normal:
                sr = sr * 255.
            sr = numpy.array(sr.cpu())
            if self.color_seq == "RGB":
                sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
            name, ext = os.path.splitext(name[0])
            save_name = self.data_name +'_x'+str(self.scale) +'_' + name+ext
            cv2.imwrite(self.save_path+save_name, sr)
            print(save_name + " Success!")
    def __pre_or_init(self, net, loader, is_normal):
        for _, data in enumerate(loader):
            img, name = data
            if "FDSR" in net.__class__.__name__:
                    sr, feat = net(img.to(self.device))
                    if self.show_freq:
                        i = 1
                        save_name = self.data_name +'_x'+str(self.scale) +'_' + os.path.splitext(name[0])[0]
                        for f in feat:
                            upsample = f.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                            upsample = ((upsample-numpy.min(upsample))/(numpy.max(upsample)-numpy.min(upsample)))*255.
                            # upsample = upsample * 255.
                            if not os.path.exists(self.save_path+save_name):
                                os.makedirs(self.save_path+save_name)
                            cv2.imwrite(self.save_path+save_name+"/"+str(i)+".png", cv2.cvtColor(upsample, cv2.COLOR_RGB2BGR))
                            i += 1
                        i = 1
                        for f in feat:
                            x_list = torch.split(f, 1, dim=1)
                            out_list = []
                            for x in x_list:
                                f = fft.fftn(x, dim=(2,3))
                                f = fft.fftshift(f, dim=(2,3))
                                out_list.append(f)
                            out = torch.cat(out_list, dim=1)
                            # out = torch.sum(out, dim=1, keepdim=False).squeeze(0).detach().numpy()
                            out = out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                            # print(out.shape)
                            out = (20*numpy.log10(1. + out))
                            out = ((out-numpy.min(out))/(numpy.max(out)-numpy.min(out)))*255.
                            out = numpy.array(out, dtype="uint8")
                            if not os.path.exists(self.save_path+save_name):
                                os.makedirs(self.save_path+save_name)
                            cv2.imwrite(self.save_path+save_name+"/freq_"+str(i)+".png", out)
                            i += 1
            else:
                sr = net(img.to(self.device))
            sr = sr.permute(0, 2, 3, 1).squeeze(0)
            if is_normal:
                sr = sr * 255.
            sr = numpy.array(sr.cpu())
            if self.color_seq == "RGB":
                sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
            name, ext = os.path.splitext(name[0])
            save_name = self.data_name +'_x'+str(self.scale) +'_' + name+ext
            cv2.imwrite(self.save_path+save_name, sr)
            print(save_name + " Success!")
    def run_demo(self):
        if self.is_pkl:
            net = torch.load(self.file_path, map_location=self.device)
        else:
            net = self.__get_model()
            para = torch.load(self.file_path, map_location=self.device)
            net.load_state_dict(para)
        net.eval()
        net.to(self.device)
        loader = utils.get_demo_loader(self.dataset, self.scale, self.color_seq, self.is_normal, self.is_input, self.is_Y)
        if self.model_mode == "post":
            if self.scale_pos == "init":
                self.__pre_or_init(net, loader, self.is_normal)
            elif self.scale_pos == "forward":
                self.__scale_pos_is_forward(net, loader, self.is_normal)
            else:
                raise NameError("WRONG MODEL SCALE POSITION!")
        elif self.model_mode == "pre":
            self.__pre_or_init(net, loader, self.is_normal)
        else:
            raise NameError("WRONG MODEL MODE!")

def demo():
    args = parse_args()
    config = load_json(args.cfg_file)
    demo = Demo(args, config)
    demo.run_demo()

if __name__ == "__main__":
    demo()