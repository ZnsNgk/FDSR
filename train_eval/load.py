import os
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Load")
    parser.add_argument("model_name", help = "model name", type = str)
    args = parser.parse_args()
    return args

def move_file(old_path, new_path):
    _, file_name = os.path.split(old_path)
    new_path = os.path.join(new_path, file_name)
    shutil.move(old_path, new_path)

def load():
    args = parse_args()
    model_name = args.model_name
    folder_list = ['./config/', './log/', 'trained_model/', 'test_result/', 'demo_output/']
    for folder in folder_list:
        file_list = os.listdir(model_name)
        if (folder == "./config/") or (folder == "./log/"):
            file = folder.replace("./", "").replace("/", "")
            ext = file.replace("config", "json")
            flag = False
            for f in file_list:
                if ext in f:
                    flag = True
                    old_path = os.path.join(model_name, f)
                    new_path = folder
                    move_file(old_path, new_path)
                    print(file+" success!")
                    break  
            if not flag:
                if ext == "json":
                    raise RuntimeError("The json file must be exist!")
                else:
                    print(file+"file not found, skip this file!")
        else:
            old_path = os.path.join(model_name, folder)
            if not os.path.exists(old_path):
                print("folder "+folder.replace("/","")+" not found, skip this folder!")
                continue
            new_path = os.path.join("./", folder)
            new_path = os.path.join(new_path, model_name)
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            file_list = os.listdir(old_path)
            for f in file_list:
                old_file = os.path.join(old_path, f)
                move_file(old_file, new_path)
            os.rmdir(old_path)
            print(folder.replace("/","")+" success!")
    os.rmdir(model_name)


if __name__ == "__main__":
    load()