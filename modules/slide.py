import os
import tiffslide as openslide
import numpy as np
import cv2
import logging
from modules import dsa

class Slide:
    def __init__(self, config, svsName) -> None:
        self.config = config
        self.svsName = svsName
        self.svsPath = f"{self.config['svsBase']}/{self.svsName}"
        self.s = openslide.open_slide(self.svsPath)

    def smoothBinary(self, mask):
        kn = 2
        iterat = 2
        kernel = np.ones((kn, kn), np.uint8) 
        for _ in range(iterat):
            mask = cv2.erode(mask, kernel, iterations=1) 
            mask = cv2.dilate(mask, kernel, iterations=1) 
        mask = cv2.dilate(mask, kernel, iterations=5) # Final Dilation Step  May Remove
        return mask

    def extractForeground(self):
        location = (0,0)
        level = self.s.level_count-1
        size = self.s.level_dimensions[self.s.level_count-1]
        lr = np.array(self.s.read_region(location, level, size))[:,:,:3]

        lrGray = cv2.cvtColor(lr, cv2.COLOR_RGB2GRAY)

        _, thre = cv2.threshold(lrGray, 220, 255, cv2.THRESH_BINARY) # THreshold for White
        threTissue = 255-thre

        self.threTissueSmooth = self.smoothBinary(threTissue)

        thresholdArea = 0.2
        contours, _ = cv2.findContours(self.threTissueSmooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        c_max = max(contours, key = cv2.contourArea)
        area_min = cv2.contourArea(c_max) * thresholdArea

        contours_filtered = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > area_min:
                contours_filtered.append(cv2.convexHull(contour, returnPoints=True))

        im_ = lrGray*0

        cv2.drawContours(im_, contours_filtered, -1, 255, -1)

        self.threTissueSmooth = np.copy(im_)
                
        return self.threTissueSmooth
    
    def extractForeground2(self):
        location = (0,0)
        level = self.s.level_count-1
        size = self.s.level_dimensions[self.s.level_count-1]
        lr = np.array(self.s.read_region(location, level, size))[:,:,:3]

        lrGray = cv2.cvtColor(lr, cv2.COLOR_RGB2GRAY)

        _, thre = cv2.threshold(lrGray, 220, 255, cv2.THRESH_BINARY) # THreshold for White
        threTissue = 255-thre

        self.threTissueSmooth = self.smoothBinary(threTissue)

        thresholdArea = 0.7
        contours, _ = cv2.findContours(self.threTissueSmooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        c_max = max(contours, key = cv2.contourArea)
        area_min = cv2.contourArea(c_max) * thresholdArea

        contours_filtered = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > area_min:
                contours_filtered.append(contour)

        im_ = lrGray*0

        cv2.drawContours(im_, contours_filtered, -1, 255, -1)

        self.threTissueSmooth = np.copy(im_)
                
        return self.threTissueSmooth

    def extractPatches(self, patchSize):
        (w, h) = self.s.level_dimensions[0]
        wn = w//patchSize
        hn = h//patchSize
        self.threTissueSmoothResized = cv2.resize(self.threTissueSmooth, (wn, hn)) # Resize Binary Mask to align with Patches

        self.patches = {}

        for ix in range(wn):
            for iy in range(hn):
                if self.threTissueSmoothResized[iy, ix]==255:
                    x = ix*patchSize
                    y = iy*patchSize
                    size = (patchSize, patchSize)
                    level = 0
                    location = (x,y)
                    self.patches[f'{x}_{y}'] = cv2.cvtColor(np.array(self.s.read_region(location, level, size))[:,:,:3], cv2.COLOR_RGB2BGR)

        return self.patches

    def createBinary(self, cnts):
        mask = np.zeros([self.s.level_dimensions[0][1], self.s.level_dimensions[0][0], 3], dtype='uint8')
        cv2.drawContours(mask, cnts, -1, [255,255,255], -1)
        self.binary = mask
        return

    def extractPairs(self, patchSize):
        (w, h) = self.s.level_dimensions[0]
        wn = w//patchSize
        hn = h//patchSize
        self.threTissueSmoothResized = cv2.resize(self.threTissueSmooth, (wn, hn)) # Resize Binary Mask to align with Patches
        
        self.patches = {}

        for ix in range(wn):
            for iy in range(hn):
                if self.threTissueSmoothResized[iy, ix]==255:
                    x = ix*patchSize
                    y = iy*patchSize
                    size = (patchSize, patchSize)
                    level = 0
                    location = (x,y)

                    im = cv2.cvtColor(np.array(self.s.read_region(location, level, size))[:,:,:3], cv2.COLOR_RGB2BGR)
                    msk = self.binary[y:y+patchSize, x:x+patchSize, :]
                    self.patches[f'{x}_{y}'] = cv2.hconcat((im, msk))
        
        return self.patches
    
    def extractPatchesInLayer(self, layername):
        item = dsa.DSAItem(self.config, self.config['itemid'], self.config['svsname'])
