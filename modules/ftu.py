import os
from abc import ABC, abstractmethod, ABCMeta
import cv2
import numpy as np

class FTU(ABC):

    __metaclass__ = ABCMeta

    @abstractmethod
    def add_feature(self, feature, value):
        """
        self.features[feature] = value
        """

    @abstractmethod
    def add_segmentation(self, layer, value):
        """
        self.segmentation[layer] = value
        """

def getCnt(mask):
    try:
        mask_2d = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    except:
        mask_2d = np.copy(mask)
    contours, _ = cv2.findContours(mask_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c_max = max(contours, key = cv2.contourArea)
    return c_max

class Tubule(FTU):
    def __init__(self, config, patch, mask, bb, elem):
        self.features = {}
        self.patch = patch
        self.mask = mask
        self.cnt = getCnt(mask)
        self.segmentation = {}
        self.bb = bb
        self.elem = elem

    def add_feature(self, feature, value):
        self.features[feature] = value
        
    def add_segmentation(self, layer, value):
        self.segmentation[layer] = value
        
    def get_segmentation(self, layer):
        return self.segmentation[layer]

    def __str__(self):
        return ', '.join(self.features)
    
    def splitIm(self, im):
        if len(im.shape)==3:
            _, w, _ = im.shape
            patch = im[:, :w//2, :]
            mask = im[:, w//2:, :]
        else:
            _, w = im.shape
            patch = im[:, :w//2]
            mask = im[:, w//2:]
        return patch, mask