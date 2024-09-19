import cv2
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
import girder_client
import shapely
from shapely import intersection
import pandas as pd
import argparse
import tiffslide as openslide
import cv2
import numpy as np

import cv2
import histomicstk as htk

import numpy as np
import scipy as sp

import skimage.io
import skimage.measure
import skimage.color

from skimage.morphology import skeletonize

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import img_as_ubyte

from shapely.geometry import Polygon, Point

from shapely import intersection
from shapely.ops import nearest_points

from statistics import mean, harmonic_mean, median, median_low, median_high, stdev, variance

from sklearn.cluster import KMeans, OPTICS
from sklearn import preprocessing

from modules.ftu import FTU

from modules.pipelines import Pipeline

from typing import List

import logging

class FeatureBuilder:
    def __init__(self):
        pass
    def build_feature(self):
        pass
    
class Segmnt(FeatureBuilder):
    
    def mask3Dlike(self, mask, patch):
        mask_3 = np.zeros_like(patch)
        mask_3[:,:,0] = mask
        mask_3[:,:,1] = mask
        mask_3[:,:,2] = mask
        return mask_3

    def getItemsInDsaFolder(self, dsaFolder):
        method = "GET"
        path = f"/item"
        r = self.gc.sendRestRequest(method, path, {'folderId': dsaFolder, 'limit': 1000})
        items = [(f['_id'], f['name']) for f in r if f['name'].endswith('.svs')]
        return items

    def getAnnotationByLayerName(self, id, name):

        method = "GET"
        path = f"/annotation"
        r = self.gc.sendRestRequest(method, path, {'itemId': id, 'limit': 1000})

        for anno in r:
            if anno['annotation']['name'] ==  name or anno['annotation']['name'] ==  f" {name}"or anno['annotation']['name'] ==  f"  {name}" :
                annoId = anno['_id']

        method = "GET"
        path = f"annotation/{annoId}"
        annotations = self.gc.sendRestRequest(method, path, {'id': annoId, 'limit': 10000000})
        return annotations

    def convertAnnotations(self, anno):
        annos = anno['annotation']['elements']
        annos_2 = []
        for a in annos:
            try:
                annos_2.append([(k[0], k[1]) for k in a['points']])
            except:
                pass
        annos_ = []
        for k in annos_2:
            if len(k)>4:
                annos_.append(shapely.Polygon(k))
        return annos_

    def computeIntersect(self, fg, bg):
        annos = []
        for anno in fg:
            for anno_bg in bg:
                try:
                    if anno.intersects(anno_bg):
                        annos.append(intersection(anno, anno_bg))
                        break
                except:
                    if anno_bg.area==0:
                        pass
        return annos

    def computeAnnosArea(self, annos):
        area = 0
        for anno in annos:
            area+= anno.area
        return area

    def splitAnnotations(self, annos):

        x_min = 1000000000
        x_max = 0
        for anno in annos:
            x = anno.centroid.coords[0][0]
            x_min = min(x_min, x)
            x_max = max(x_max, x)
        x_threshold = ( x_max - x_min ) / 2

        left = []
        right = []

        for anno in annos:
            x = anno.centroid.coords[0][0]
            if x<x_threshold:
                left.append(anno)
            else:
                right.append(anno)

        return left, right


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

    def smoothBinary(self, mask):
        kn = 5
        iterat = 5
        kernel = np.ones((kn, kn), np.uint8) 
        for _ in range(iterat):
            mask = cv2.erode(mask, kernel, iterations=1) 
            mask = cv2.dilate(mask, kernel, iterations=1) 
        return mask

    def removeSmall(self, mask):
        threshold_area = 0.3
        mask_2d = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(mask_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        c_max = max(contours, key = cv2.contourArea)
        area_min = cv2.contourArea(c_max) * threshold_area

        contours_filtered = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > area_min:
                contours_filtered.append(contour)
        blank_image = np.zeros(mask.shape, np.uint8)
        cv2.fillPoly(blank_image, pts=contours_filtered, color= (255,255,255))

        return blank_image


    def applyThreshold(self, patch, opmode, msk_orig):
        kn = 5
        iterat = 5
        try:
            msk = cv2.cvtColor(msk_orig, cv2.COLOR_BGR2GRAY)
            if opmode=="lf":
                white_threshold_sat = 15
                hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                lower_white = np.array([0,0,0], dtype=np.uint8)
                upper_white = np.array([255,white_threshold_sat,255], dtype=np.uint8)
                mask = cv2.inRange(hsv, lower_white, upper_white)
                kernel = np.ones((kn, kn), np.uint8) 
                for _ in range(iterat):
                    mask = cv2.erode(mask, kernel, iterations=1) 
                    mask = cv2.dilate(mask, kernel, iterations=1)
                return self.mask3Dlike(mask*msk, patch)  
            elif opmode=="tbm":
                hue_min = int(295 * 180/360)
                hue_max = int(315 * 180/360)
                val_min = 0
                val_max = 255
                hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                lower_white = np.array([hue_min,0,val_min], dtype=np.uint8)
                upper_white = np.array([hue_max,255,val_max], dtype=np.uint8)
                mask = cv2.inRange(hsv, lower_white, upper_white)
                return self.mask3Dlike(mask*msk, patch)   
            elif opmode=="epithelium":
                mask_lf = self.applyThreshold(patch, "lf", msk_orig)
                mask_tbm = self.applyThreshold(patch, "tbm", msk_orig)
                return (msk_orig*255)-mask_tbm-mask_lf
        except:
            print("Except inside apply threshold")  

    def frmtCnt(self, cnt):
        cntrrr = np.vstack(cnt).squeeze()  
        n, _ = cntrrr.shape
        cntrrrLst = []
        for u in range(n):
            cntrrrLst.append([cntrrr[u, 0], cntrrr[u, 1]])
        return cntrrrLst
    
    def segmentSingleTubule(self, config, patch, mask):

        stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
        #print('stain_color_map:', stain_color_map, sep='\n')

        # specify stains of input image
        stains = ['hematoxylin',  # nuclei stain
                  'eosin',        # cytoplasm stain
                  'null']         # set to null if input contains only two stains

        # create stain matrix
        W = np.array([stain_color_map[st] for st in stains]).T

        # perform standard color deconvolution
        imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(patch, W)

        ## Thresholding
        eosinStain = imDeconvolved.Stains[:, :, 1]

        imInput_cv = img_as_ubyte(patch)
        eosinStain_cv = img_as_ubyte(eosinStain)

        eosin = np.copy(eosinStain_cv)

        _, thre = cv2.threshold(eosin, 200, 255, cv2.THRESH_BINARY_INV)

        eosin_color = cv2.cvtColor(eosin, cv2.COLOR_GRAY2BGR)
        binary_output = cv2.cvtColor(thre, cv2.COLOR_GRAY2BGR)

        binary_output_smth = self.smoothBinary(binary_output)
        binary_output_rm = self.removeSmall(binary_output_smth)

        lf_mask = self.applyThreshold(patch, "lf", mask)
        
        #print("lf_mask")
        #print(lf_mask.shape)
        
        lf_mask2 = self.removeSmall(lf_mask*255)

        kernel = np.ones((9, 9), np.uint8) 
        mask_ = cv2.dilate(mask, kernel, iterations=1) 
        
        #print(np.max(mask_))
        #print(np.max(lf_mask2))
        #print(np.max(binary_output_rm))
        
        #plt.imshow(mask_)
        #plt.show()
        
        #plt.imshow(lf_mask2)
        #plt.show()
        
        #plt.imshow(binary_output_rm)
        #plt.show()

        epithelium = (mask_ -lf_mask2 -binary_output_rm)   # Tubule - Lumen - TBM
        epithelium_tuned = self.removeSmall(epithelium)
        
        #plt.imshow(epithelium)
        #plt.show()

        negativeEpithelium = np.array((255 - epithelium)*(mask_/255), dtype='uint8')
        
        #plt.imshow(negativeEpithelium)
        #plt.show()
        
        negativeEpitheliumTuned = self.removeSmall(negativeEpithelium)

        epithelium_tuned_again = epithelium_tuned - negativeEpitheliumTuned

        stain_color_map

        patchGray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patchGray3 = np.zeros_like(patch, dtype='uint8')
        patchGray3[:,:,0] = patchGray; patchGray3[:,:,1] = patchGray; patchGray3[:,:,2] = patchGray; 

        d1 = patch   # H&E
        self.tubule.add_segmentation('deconv', np.array(eosin_color * (mask_/255), dtype='uint8'))    # Deconvolution

        self.tubule.add_segmentation('tbm', np.array(binary_output_rm * (mask_/255), dtype='uint8'))  # TBM binary
        self.tubule.add_segmentation('lmn', np.array(lf_mask2 * (mask_/255), dtype='uint8'))  # Lumen binary

        d5 = np.array( (epithelium)*(mask_/255), dtype='uint8')
        self.tubule.add_segmentation('epm', np.array( (epithelium_tuned_again)*(mask_/255), dtype='uint8')) # Epithelium binary

        d7 = patchGray3    # Just gray scale
        
        return
                
    def build_feature(self, config, ftu):
        try:
            self.tubule
        except:
            self.tubule = ftu
        
        print("Segmentation logic")
        self.segmentSingleTubule(config, self.tubule.patch, self.tubule.mask)
        
        plt.imshow(self.tubule.get_segmentation('deconv'))
        plt.show()
        
        plt.imshow(self.tubule.get_segmentation('tbm'))
        plt.show()
        
        plt.imshow(self.tubule.get_segmentation('lmn'))
        plt.show()

        plt.imshow(self.tubule.get_segmentation('epm'))
        plt.show()
        
        return
        
class EpmFtr(FeatureBuilder):
    
    def frmtCnt(self, cnt):
        cntrrr = np.vstack(cnt).squeeze()  
        n, _ = cntrrr.shape
        cntrrrLst = []
        for u in range(n):
            cntrrrLst.append([cntrrr[u, 0], cntrrr[u, 1]])
        return cntrrrLst
    
    def measureEpithelium(self, config, ept_msk, mask_):
        d = np.array((ept_msk)*(mask_/255), dtype='uint8')
        mask_2d = cv2.cvtColor(ept_msk, cv2.COLOR_BGR2GRAY)

        _, thre = cv2.threshold(mask_2d, 20, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(d, [contours[-1]], -1, (0, 255, 0), 3) 

        cntrrrLst = self.frmtCnt(contours[-1])

        outerCnt = Polygon(cntrrrLst)

        _, threInv = cv2.threshold(255-mask_2d, 20, 255, cv2.THRESH_BINARY)
        contoursInv, _ = cv2.findContours(threInv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        innerCnts = [Polygon(self.frmtCnt(k)) for k in contoursInv]

        cnts = []
        innerCntLst = []
        for j in range(len(innerCnts)):
            innerCnt = innerCnts[j]
            if intersection(innerCnt, outerCnt).area == innerCnt.area:
                cnts.append(contoursInv[j])
                innerCntLst.append(self.frmtCnt(contoursInv[j]))

        #### cv2.drawContours(d, cnts, -1, (0, 0, 255), 3) 

        otrPntPair = tuple(outerCnt.exterior.coords)

        innerPolys = [ Polygon(self.frmtCnt(k)) for k in cnts ]

        distances = []

        for a in range(len(otrPntPair)):
            if a%10==0:
                otrPnt = otrPntPair[a]
                outerPnt = Point(otrPnt)
                innerPntPairs = []
                for z in range(len(innerPolys)):
                    innerPoly = innerPolys[z]
                    innerPntPairs.extend(tuple(innerPoly.exterior.coords))
                dst = []
                for j in range(len(innerPntPairs)):
                    inner = innerPntPairs[j]
                    innerPnt = Point(inner)
                    ds = outerPnt.distance(innerPnt)
                    dst.append(ds)
                try:
                    mn = min(dst)
                    distances.append(mn)
                    idx = dst.index(mn)
                    innerMin = innerPntPairs[idx]
                    innerPntMin = Point(innerMin)
                    ####  cv2.line(d, (int(innerPntMin.x), int(innerPntMin.y)), (int(outerPnt.x), int(outerPnt.y)), (200, 190, 180), thickness=1)
                except:
                    pass
            pass

        self.tubule.add_feature('epm_mean', mean(distances)*config['spatialResolution'])
        self.tubule.add_feature('epm_harmonic_mean', harmonic_mean(distances)*config['spatialResolution'])
        self.tubule.add_feature('epm_median', median(distances)*config['spatialResolution'])
        self.tubule.add_feature('epm_median_low', median_low(distances)*config['spatialResolution'])
        self.tubule.add_feature('epm_median_high', median_high(distances)*config['spatialResolution'])
        self.tubule.add_feature('epm_stdev', stdev(distances)*config['spatialResolution'])
        self.tubule.add_feature('epm_variance', variance(distances)*config['spatialResolution']*config['spatialResolution'])

        return
    
    def build_feature(self, config, ftu):
        try:
            self.tubule
        except:
            self.tubule = ftu
        feature = "Epithelium Thickness"
        self.measureEpithelium(config, self.tubule.get_segmentation('epm'), self.tubule.mask)

class LmnFtr(FeatureBuilder):

    def measureLumen(self, config, lmn_msk, mask_):

        def getContourMax(mmm):
            mmm2 = cv2.cvtColor(mmm, cv2.COLOR_BGR2GRAY)
            _, thre = cv2.threshold(mmm2, 20, 255, cv2.THRESH_BINARY)
            ccc, _ = cv2.findContours(thre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt = max(ccc, key = len)
            return cnt

        c1 = getContourMax(lmn_msk) 
        c2 = getContourMax(mask_) 

        a1 = cv2.contourArea(c1)
        a2 = cv2.contourArea(c2)

        x,y,w,h = cv2.boundingRect(c1)
        aspect_ratio = float(w)/h

        hull = cv2.convexHull(c1, returnPoints=False)
        hull_ = cv2.convexHull(c1)
        hull_area = cv2.contourArea(hull_)
        solidity = float(a1)/hull_area

        equi_diameter = np.sqrt(4*a1/np.pi)

        a1a2 = a1/a2

        (x,y),(MA,ma),angle = cv2.fitEllipse(c1)

        convexityDefectsHull = cv2.convexityDefects(c1, hull)

        defectsHull = [f[-1] for f in [list(k[0]) for k in list(convexityDefectsHull)]]

        defectsHull_mean = mean(defectsHull)
        defectsHull_std = stdev(defectsHull)

        self.tubule.add_feature('lmn_area', a1 * (config['spatialResolution']*config['spatialResolution']))
        self.tubule.add_feature('lmn_area_fraction', a1a2)
        self.tubule.add_feature('lmn_solidity', solidity)
        self.tubule.add_feature('lmn_hull_area', hull_area * (config['spatialResolution']*config['spatialResolution']))
        self.tubule.add_feature('lmn_aspect_ratio', aspect_ratio)
        self.tubule.add_feature('lmn_equi_diameter', equi_diameter*config['spatialResolution'])
        self.tubule.add_feature('lmn_angle', angle)
        self.tubule.add_feature('lmn_defectsHull_mean', defectsHull_mean)
        self.tubule.add_feature('lmn_defectsHull_std', defectsHull_std)

        return

    def build_feature(self, config, ftu):
        feature = "Luminal Fraction"
        try:
            self.tubule
        except:
            self.tubule = ftu
        self.measureLumen(config, self.tubule.get_segmentation('lmn'), self.tubule.mask)


class BscFtr(FeatureBuilder):

    def measureBsc(self, config, mask_):

        def getContourMax(mmm):
            mmm2 = cv2.cvtColor(mmm, cv2.COLOR_BGR2GRAY)
            _, thre = cv2.threshold(mmm2, 20, 255, cv2.THRESH_BINARY)
            ccc, _ = cv2.findContours(thre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt = max(ccc, key = len)
            return cnt

        c2 = getContourMax(mask_) 

        a2 = cv2.contourArea(c2) 

        x,y,w,h = cv2.boundingRect(c2)
        aspect_ratio = float(w)/h

        hull = cv2.convexHull(c2, returnPoints=False)
        hull_ = cv2.convexHull(c2)
        hull_area = cv2.contourArea(hull_)
        solidity = float(a2)/hull_area

        equi_diameter = np.sqrt(4*a2/np.pi)

        perimeter = cv2.arcLength(c2, True)

        circularity = 4*np.pi*a2/(perimeter*perimeter)

        self.tubule.add_feature('bsc_area', a2 * (config['spatialResolution']*config['spatialResolution']))
        self.tubule.add_feature('bsc_circularity', solidity)
        #self.tubule.add_feature('bsc_solidity', solidity)
        #self.tubule.add_feature('bsc_hull_area', hull_area)
        self.tubule.add_feature('bsc_aspect_ratio', aspect_ratio)
        #self.tubule.add_feature('bsc_equi_diameter', equi_diameter)

        return

    def build_feature(self, config, ftu):
        feature = "Basic Features of Tubules"
        try:
            self.tubule
        except:
            self.tubule = ftu
        self.measureBsc(config, self.tubule.mask)

class TbmFtr(FeatureBuilder):

    def frmtCnt(self, cnt):
        cntrrr = np.vstack(cnt).squeeze()  
        n, _ = cntrrr.shape
        cntrrrLst = []
        for u in range(n):
            cntrrrLst.append([cntrrr[u, 0], cntrrr[u, 1]])
        return cntrrrLst
    
    def measureTbm(self, config, ept_msk, mask_):

        ept_msk = cv2.cvtColor(ept_msk, cv2.COLOR_BGR2GRAY)

        _, binary_mask = cv2.threshold(ept_msk, 127, 255, cv2.THRESH_BINARY)

        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 3)
        
        skeleton = skeletonize(binary_mask)

        sk = list(np.transpose((skeleton==True).nonzero()))
        distances = [dist_transform[a[0],a[1]]*2 for a in sk]

        self.tubule.add_feature('tbm_mean', mean(distances)*config['spatialResolution'])
        self.tubule.add_feature('tbm_harmonic_mean', harmonic_mean(distances)*config['spatialResolution'])
        self.tubule.add_feature('tbm_median', median(distances)*config['spatialResolution'])
        self.tubule.add_feature('tbm_median_low', median_low(distances)*config['spatialResolution'])
        self.tubule.add_feature('tbm_median_high', median_high(distances)*config['spatialResolution'])
        self.tubule.add_feature('tbm_stdev', stdev(distances)*config['spatialResolution'])
        self.tubule.add_feature('tbm_variance', variance(distances)*config['spatialResolution']*config['spatialResolution'])

        return

    def build_feature(self, config, ftu):
        feature = "TBM Thickness"
        try:
            self.tubule
        except:
            self.tubule = ftu
        self.measureTbm(config, self.tubule.get_segmentation('tbm'), self.tubule.mask)

class TbmFtrWoMask(FeatureBuilder):

    def frmtCnt(self, cnt):
        cntrrr = np.vstack(cnt).squeeze()  
        n, _ = cntrrr.shape
        cntrrrLst = []
        for u in range(n):
            cntrrrLst.append([cntrrr[u, 0], cntrrr[u, 1]])
        return cntrrrLst
    
    def measureTbm(self, config, patch):

        binary_mask = self.applyThreshold(patch)

        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 3)
        
        skeleton = skeletonize(binary_mask)

        sk = list(np.transpose((skeleton==True).nonzero()))
        distances = [dist_transform[a[0],a[1]]*2 for a in sk]

        self.tubule.add_feature('tbm_mean', mean(distances)*config['spatialResolution'])
        self.tubule.add_feature('tbm_harmonic_mean', harmonic_mean(distances)*config['spatialResolution'])
        self.tubule.add_feature('tbm_median', median(distances)*config['spatialResolution'])
        self.tubule.add_feature('tbm_median_low', median_low(distances)*config['spatialResolution'])
        self.tubule.add_feature('tbm_median_high', median_high(distances)*config['spatialResolution'])
        self.tubule.add_feature('tbm_stdev', stdev(distances)*config['spatialResolution'])
        self.tubule.add_feature('tbm_variance', variance(distances)*config['spatialResolution']*config['spatialResolution'])

        return

    def applyThreshold(self, patch):
        try:
            hue_min = int(295 * 180/360)
            hue_max = int(315 * 180/360)
            val_min = 0
            val_max = 255
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            lower_white = np.array([hue_min,0,val_min], dtype=np.uint8)
            upper_white = np.array([hue_max,255,val_max], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_white, upper_white)
            return self.mask3Dlike(mask, patch)   
        except:
            print("Except inside apply threshold")  

    def build_feature(self, config, ftu):
        feature = "TBM Thickness"
        try:
            self.tubule
        except:
            self.tubule = ftu
        self.measureTbm(config, self.patch)

class NaglahPipeline(Pipeline):
    def __init__(self, config, builders):
        self.name = config["name"]
        self.pipeline = "NaglahPipeline"
        self.builders = builders

    def run(self, config, ftu):
        logging.warning(f"Pipeline {self.pipeline} is starting")
        for builder in self.builders:
            builder.build_feature(config, ftu)