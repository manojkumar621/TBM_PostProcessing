from __future__ import print_function, division
from torch.utils.data import Dataset
from skimage import io
import glob
import os
import torch
from torchvision import transforms
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """Dataset for inference with raw images only"""

    # def __init__(self, foldername, hp, train=True, transform=None):
    def __init__(self, hp, train=True, transform=None):
        """Args:
            hp: Hyperparameters object with paths
            train (bool): Whether to use training data or not
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.train = train
        self.path = hp.train if train else hp.valid
        self.image_list = glob.glob(
            os.path.join(self.path, "input_crop", "*.[jp][pn]g"), recursive=True
        )
        # self.image_list = glob.glob(
        #     os.path.join(self.path, f'{foldername}', "*.png"), recursive=True
        # )
        valid_dir = os.path.join(self.path, "input_crop")
        print(f"Found {len(self.image_list)} images in {valid_dir}")
        # logger.info(f"Found {len(self.image_list)} images in {foldername}")
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        image = io.imread(image_path)

        sample = {"sat_img": image, "image_path": image_path}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def getPath(self, idx):
        image_path = self.image_list[idx]
        return image_path

class ToTensorTarget(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sat_img = sample["sat_img"]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        return {
            "sat_img": transforms.functional.to_tensor(sat_img),
            "image_path": sample["image_path"]
        }
