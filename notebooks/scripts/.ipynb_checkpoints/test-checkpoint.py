import os, sys
notebook_dir = os.getcwd()
project_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
if project_dir not in sys.path:
    sys.path.append(project_dir)
utils_dir = os.path.abspath(os.path.join(notebook_dir, '..', 'utils'))
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

import warnings
warnings.simplefilter("ignore", (UserWarning, FutureWarning))
from hparams import HParam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import dataloader
import metrics
from core.res_unet import ResUnet
from core.res_unet_plus import ResUnetPlusPlus
from logger import MyWriter
import torch
import argparse
import glob
import skimage
from torch import nn
from torchvision.utils import save_image
from logging_setup import log_execution_time
import logging
import traceback

def infer(hp):
    try:
        name = hp.name
        output_dir = hp.pred
        os.makedirs("{}/{}".format(hp.log, name), exist_ok=True)
        writer = MyWriter("{}/{}".format(hp.log, name))

        if hp.RESNET_PLUS_PLUS:
            model = ResUnetPlusPlus(3).cuda()
            # model = nn.DataParallel(model)
        else:
            model = ResUnet(3, 64).cuda()

        checkpoint = torch.load(hp.checkpoints)
        model.load_state_dict(checkpoint["state_dict"])
        logging.info("=> loaded checkpoint '{}' (epoch {})".format(hp.checkpoints, checkpoint["epoch"]))

        folder_names = [name for name in os.listdir(hp.valid) if os.path.isdir(os.path.join(hp.valid, name))]
        print(folder_names)
        print (f'There are {len(folder_names)} image folders')
        for foldername in folder_names:
            output_dir = hp.pred
            output_dir = os.path.join(output_dir, foldername)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            mass_dataset_val = dataloader.ImageDataset(
                foldername, hp, False, transform=transforms.Compose([dataloader.ToTensorTarget()])
            )
            print(f'working on {foldername}')

            val_dataloader = DataLoader(
                mass_dataset_val, batch_size=1, num_workers=2, shuffle=False
            )

            model.eval()
            # print(type(val_dataloader))

            for idx, data in enumerate(tqdm(val_dataloader, desc="validation")):
                # print(data)
                inputs = data["sat_img"].cuda()
                prob_map = model(inputs) # last activation was a sigmoid
                outputs = (prob_map > 0.3).float()
                # print(outputs.shape)
                image_name = mass_dataset_val.getPath(idx).split("/")[-1]
                print(image_name)
                img = outputs[0] #torch.Size([3,28,28]
                # save_image(img, f'{output_dir}/pred_{idx}_{image_name}')
                save_image(img, f'{output_dir}/{image_name}')
    except Exception as e:
        print(f'Exception while infering files: {str(e)}')
        logging.error(f'Exception while infering files:{str(e)}')
        logging.error("Traceback: " + traceback.format_exc())
        
# if __name__ == "__main__":
#     args = {
#         "name": "ResUnetTrain3",
#         "config": "/home/manojkumargalla/PostProcess/config/default3.yaml",
#         "resume": ""
#     }
    
#     class Struct:
#         def __init__(self, entries):
#             self.__dict__.update(entries)
            
#     args = Struct(args)
    
#     hp = HParam(args.config)
#     with open(args.config, "r") as f:
#         hp_str = "".join(f.readlines())

#     main(hp)
    






