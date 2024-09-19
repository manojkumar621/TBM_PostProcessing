import os, sys
notebook_dir = os.getcwd()
project_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
if project_dir not in sys.path:
    sys.path.append(project_dir)
modules_dir = os.path.abspath(os.path.join(notebook_dir, '..', 'modules'))
if modules_dir not in sys.path:
    sys.path.append(modules_dir)
import cv2
import numpy as np
from slide import Slide
from dsa import DSAFolder, DSAItem, DSA
import json
import glob
from logging_setup import log_execution_time
import logging
import traceback

def check():
    print('checked')

def makeDir(mydir):
    if not os.path.exists(mydir):
        os.mkdir(mydir)

def generateAnno(contoursColors, predicted = True):

    annos = []
    elems = []

    for q in range(len(contoursColors)):
        cnts = contoursColors[q]['cnt']
        color = contoursColors[q]['color']

        for f in range(len(cnts)):
            cnt = cnts[f][0]
            hie = cnts[f][1]
            # print(hie)
            if hie[3]==-1:
                points = []
                for i in cnt:
                    for j in i:
                        x = int(j[0])
                        y = int(j[1])
                        points.append([x, y, 0])
    
                elem = {
                    "type": "polyline",
                    "lineColor": f"rgb({color})", 
                    "lineWidth": 3, 
                    "fillColor": f"rgba({color}, 0.5)",
                    "points": None,
                    "closed": True
                }

                elem["points"] = points

                if len(points)>1:
                    if (f+1)<len(cnts) and cnts[f+1][1][3]==-1: #No Children
                        pass
                    else: # Get Children
                        z = f+1
                        holes = []
                        while True:
                            hole = []
                            try:
                                cntChild = cnts[z][0]
                                hieChild = cnts[z][1]
                            except:
                                break
                            if hieChild[3]==-1:
                                break
                            for i in cntChild:
                                for j in i:
                                    x = int(j[0])
                                    y = int(j[1])
                                    hole.append([x, y, 0])
                            holes.append(hole)
                            z+=1
                        elem["holes"] = holes
                elems.append(elem)
    if predicted:
        annos_name = 'newmodel_predicted'
    else:
        annos_name = 'newmodel_filtered'
    anno = {
        "name": annos_name, 
        "description": 'tbm_ai',  
        "elements": None                        
    }
    anno["elements"] = elems
    annos.append(anno)

    return annos



def getContours(mask):

    def smoothBinary(mask2d):
        kn = 2
        iterat = 2
        kernel = np.ones((kn, kn), np.uint8) 
        for _ in range(iterat):
            mask2d = cv2.erode(mask2d, kernel, iterations=1) 
            mask2d = cv2.dilate(mask2d, kernel, iterations=1) 
        return mask2d
    
    mask2 = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    maskSmooth = smoothBinary(mask2)

    thresholdArea = 0.0001

    contours, hierarchy = cv2.findContours(maskSmooth, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"found {len(contours)} contours")

    c_max = max(contours, key = cv2.contourArea)
    area_min = cv2.contourArea(c_max) * thresholdArea
    
    print(len(contours))
    
    cntheis = sorted(zip(contours, hierarchy[0]), key=lambda b: cv2.contourArea(b[0]) , reverse=True)
    
    mxmx = int(1.0 * len(contours))
    
    contours_filtered = []
    
    for qq in range(mxmx):
        # if qq%50==0:
            # print(f"processing {qq}/{mxmx}")
        contour = cntheis[qq][0]
        hie = cntheis[qq][1]
        contours_filtered.append((contour,hie))

    return contours_filtered

def save(tbm_config, predicted = True):
    try:
        if predicted:
            tiles_folder = tbm_config.pred
        else:
            tiles_folder = tbm_config.filtered_tiles
        filenames = glob.glob(f'{tiles_folder}/*/')
        for filename in filenames:
            files = os.listdir(f'{filename}')
            fname = filename.split('/')[-2]
            print(f'processing file - {filename}')
            logging.info(f'processing file - {filename}')
            # print(files)
            xx = int(files[0].split('_')[0])
            yy = int(files[0].split('_')[1])
            mask = np.zeros((yy, xx, 3), dtype='uint8')
            for file in files:
                x = int(file.split('_')[2].strip('x'))
                y = int(file.split('_')[3].split('.')[0].strip('y'))
                w, h = 512, 512
                img = cv2.imread( f"{filename}/{file}",3)
                h, w, _ = img.shape
                if y + h <= yy and x + w <= xx:
                    mask[y:y+h, x:x+w, :] = img
                else:
                    print(f"Skipping tile {file} due to out-of-bounds placement.")
                    print(f'y + h is {y+h}, x + w is {x+w}')

            contours = getContours(mask)
            contoursColors = [{'cnt': contours, 'color': '255, 20, 20'}]

            file_name = filename.split('/')[-2]
            annotations_filename = f'{file_name}_tbm_test.json'
            if predicted:
                json_dir = tbm_config.predicted_annotations_path
            else:
                json_dir = tbm_config.filtered_annotations_path
            makeDir(json_dir)
            if os.path.exists(f'{json_dir}/{annotations_filename}'):
                print(f'{file_name} exists! Hence, continuing')
                continue

            annos = generateAnno(contoursColors, predicted)
            with open(f'{json_dir}/{annotations_filename}', 'w') as outfile:
                json.dump(annos, outfile)
    except Exception as e:
        print(f'Exception while Saving annotations: {str(e)}')
        logging.error(f'Exception while Saving annotations:{str(e)}')
        logging.error("Traceback: " + traceback.format_exc())


        
        



