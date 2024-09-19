import cv2

import girder_client
import shapely
import tiffslide as openslide
import numpy as np
import os
import logging
import json

from modules.ftu import FTU, Tubule

import numpy as np

import argparse

class DSA:
    def __init__(self, config) -> None:
        self.config = config
        apiUrl = config['apiUrl']
        username = config['username']
        password = config['password']
        self.gc = girder_client.GirderClient(apiUrl=apiUrl)
        self.gc.authenticate(username, password)

    def postAnno(self, fid, annos):
        for anno in annos:
            anno = json.dumps(anno)
            method = "POST"
            path = "/annotation"
            r = self.gc.sendRestRequest(method, path, parameters={'itemId': fid}, data=anno)
        return

class DSAFolder(DSA):
    def __init__(self, config) -> None:
        super().__init__(config)
        folderid = config['fid']
        self.items = self.getItemsInDsaFolder(folderid)

    def getItemsInDsaFolder(self, dsaFolder):
        method = "GET"
        path = f"/item"
        r = self.gc.sendRestRequest(method, path, {'folderId': dsaFolder, 'limit': 1000})
        items = [(f['_id'], f['name']) for f in r if f['name'].endswith('.svs')]
        return items

class DSAItem(DSA):
    def __init__(self, config, fid, svsname) -> None:
        super().__init__(config)
        self.layerName = config['layerName']
        self.fid = fid
        self.svsname = svsname
        self.name = svsname.replace('.svs', '')
        self.svsBase = config['svsBase']
        self.annotationsShapely = self.convertAnnotations(self.getAnnotationByLayerName(self.layerName))
        self.annoRow = self.getAnnotationByLayerName(self.layerName)
        self.annoElems = self.annoRow['annotation']['elements']
        self.annotations = self.convertAnnotationsCV(self.getAnnotationByLayerName(self.layerName))
        self.s = openslide.open_slide(f"{self.svsBase}/{self.svsname}")

    def getAnnotationByLayerName(self, name):

        method = "GET"
        path = f"/annotation"
        r = self.gc.sendRestRequest(method, path, {'itemId': self.fid, 'limit': 1000})

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
    
    def convertAnnotationsCV(self, anno):
        annos = anno['annotation']['elements']
        annos_2 = []
        for a in annos:
            annos_2.append(np.array([[k[0], k[1]] for k in a['points']], dtype=np.int32))
        return annos_2
    
    def extract(self):
        outputdir = self.config['outputdir']

        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
        
        for i in range(len(self.annotations)):
            try:
                patch, mask = self.getPatchMask(i)
                if self.config['masked']:
                    out = np.array(patch*(mask/255), dtype='uint8')
                else:
                    out = np.copy(patch)
                cv2.imwrite(f'{outputdir}/{self.name}_{i}_{self.x}_{self.y}.jpg', out)
            except:
                pass

    def getTubules(self):

        self.ftus = []
        
        for i in range(len(self.annotations)):
            patch, mask, bb = self.getPatchMask(i)
            self.ftus.append(Tubule({}, patch, mask, bb, self.annoElems[i]))

        return self.ftus

    def extractFixed(self, w, h):
        outputdir = self.config['outputdir']

        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
        
        for i in range(len(self.annotations)):
            try:
                patch, mask = self.getPatchMaskFixed(i, w, h)
                if self.config['masked']:
                    out = np.array(patch*(mask/255), dtype='uint8')
                else:
                    out = np.copy(patch)
                cv2.imwrite(f'{outputdir}/{self.name}_{i}_{self.x}_{self.y}.jpg', out)
            except:
                pass

    def getMaxBB(self):
        
        w_max = 0
        h_max = 0
        for i in range(len(self.annotations)):
            try:
                w, h = self.getWH(i)
                w_max = max(w, w_max)
                h_max = max(h, h_max)
            except:
                pass
        return w_max, h_max

    def getWH(self, tid):
        polygon1 = self.annotations[tid]
        bb = polygon1.bounds
        w = int(bb[2]-bb[0])
        h = int(bb[3]-bb[1])
        
        return w, h

    def getPatchMask(self, tid):
        polygon1 = self.annotationsShapely[tid]
        x,y = polygon1.exterior.xy
        bb = polygon1.bounds

        size_1 = [int(bb[3]-bb[1]), int(bb[2]-bb[0])]

        mask = np.zeros(size_1)

        points = [[int(x-bb[0]), int(y-bb[1])] for x, y in zip(*polygon1.boundary.coords.xy)]

        mask = cv2.fillPoly(mask, np.array([points]).astype(np.int32), color=255)

        size_2 = [int(bb[2]-bb[0]), int(bb[3]-bb[1])]
        location = (int(bb[0]), int(bb[1]))

        self.x = int(bb[0])
        self.y = int(bb[1])
        self.w = int(bb[2]-bb[0])
        self.h = int(bb[3]-bb[1])
        self.polygon = polygon1

        level = 0

        patch = np.array(self.s.read_region(location, level, size_2))[:,:,:3]
        mask = cv2.cvtColor(np.array(mask, dtype='uint8'), cv2.COLOR_GRAY2RGB)
        
        return patch, mask, {'x':self.x, 'y':self.y, 'w':self.w, 'h':self.h}
    
    def getPatchMaskFixed(self, tid, w, h):
        polygon1 = self.annotations[tid]
        x,y = polygon1.exterior.xy
        bb = polygon1.bounds

        size_2 = [int(bb[2]-bb[0]), int(bb[3]-bb[1])]

        wd = max((w - int(bb[2]-bb[0]))//2, 0)
        hd = max((h - int(bb[3]-bb[1]))//2, 0)

        size_2 = [int(bb[2]-bb[0])+2*wd, int(bb[3]-bb[1])+2*hd]

        location = (int(bb[0])-wd, int(bb[1])-hd)

        self.x = int(bb[0])-wd
        self.y = int(bb[1])-hd

        level = 0

        patch = np.array(self.s.read_region(location, level, size_2))[:,:,:3]

        size_1 = [int(bb[3]-bb[1])+2*hd, int(bb[2]-bb[0])+2*wd]

        mask = np.zeros(size_1)

        points = [[int(x-bb[0]+wd), int(y-bb[1]+hd)] for x, y in zip(*polygon1.boundary.coords.xy)]

        mask = cv2.fillPoly(mask, np.array([points]).astype(np.int32), color=255)

        mask = cv2.cvtColor(np.array(mask, dtype='uint8'), cv2.COLOR_GRAY2RGB)
        
        return patch, mask
    
    def getPatchesInLayer(self, layername, patchSize):
        anno = self.getAnnotationByLayerName(self, layername) # Get ROI
        annoShapely = self.convertAnnotations(anno) # Convert to Shapely Format

        def createRectangle(bb):
            return shapely.Polygon([[bb[0], bb[1]], [bb[0]+bb[2], bb[1]], [bb[0]+bb[2], bb[1]+bb[3]], [bb[0], bb[1]+bb[3]], [bb[0], bb[1]]])

        patches = {}
        for i in range(len(annoShapely)):
            a = annoShapely[i]
            bb = a.bounds
            x = bb[0]
            y = bb[1]
            w = bb[2]
            h = bb[3]

            stepsW = w//patchSize
            stepH = h//patchSize

            for qw in range(stepsW):
                for qh in range(stepH):
                    xx = x + qw * patchSize
                    yy = y + qh * patchSize
                    ww = patchSize
                    hh = patchSize
                    p = createRectangle([xx, yy, ww, hh])
                    intersection = a.intersects(p)
                    if intersection:
                        location = (xx, yy)
                        level = 0
                        size = (patchSize, patchSize)
                        
                        patch = np.array(self.s.read_region(location, level, size))[:,:,:3]
                        patches[f'{layername}_{xx}_{yy}'] = patch