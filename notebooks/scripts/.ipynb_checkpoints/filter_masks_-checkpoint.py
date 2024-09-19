import os

os.environ["CV_IO_MAX_IMAGE_PIXELS"] = f"{2**99}"
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = f"{2**63}"

from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
import tiffslide
import gc
from logging_setup import log_execution_time
import logging
import traceback

def makeDir(mydir):
    if not os.path.exists(mydir):
        os.mkdir(mydir)
def get_binary_mask(wsi_path, annotations_path):
    with open(annotations_path) as f:
        annotations = json.load(f)
    # Load the WSI file
    slide = tiffslide.open_slide(wsi_path)
    dimensions = slide.dimensions
    width, height = dimensions
    
    binary_mask = np.zeros((height, width), dtype=np.uint8)
    if "annotation" in annotations:
        elements = annotations['annotation']['elements']
    else:
        elements = annotations[0]['elements']
    for element in elements:
        if element['type'] == 'polyline':
            points = element['points']
            # print(len(points[0]))
            if element['closed'] and len(points) > 0 and len(points[0]) == 3:
                points = np.array(points, dtype=np.int32)
                # Ensure the points are in the correct shape (N, 1, 2)
                points = np.array([[p[0], p[1]] for p in points], dtype=np.int32)
                points = points.reshape((-1, 1, 2))
                contours = [points]
                cv2.drawContours(binary_mask, contours, -1, color=255, thickness=cv2.FILLED)  # Draw the filled contour

    slide.close()
    # print('Done!')
    return binary_mask
    


def save_image_masks(wsi, WSI_images_dir, tbm_annotations_dir, tubules_annotations_dir, binaries_dir):
    wsi_path = os.path.join(WSI_images_dir, wsi)
    wsi_name = str(wsi.split('.')[0])
    print(f'Processing image - {wsi_name}')
    tbm_annotations_path = os.path.join(tbm_annotations_dir, f'{wsi_name}_tbm_test.json')
    # tubules_annotations_path = os.path.join(tubules_annotations_dir, f'{wsi_name}_tubules.json')
    tubules_annotations_path = os.path.join(tubules_annotations_dir, f'{wsi_name}.json')
    print('Generating binary masks')
    binary_mask_of_tubules = get_binary_mask(wsi_path, tubules_annotations_path)
    binary_mask_of_tbms = get_binary_mask(wsi_path, tbm_annotations_path)
    binary_path = os.path.join(binaries_dir, f'{wsi_name}')
    if not os.path.exists(binary_path):
        os.mkdir(binary_path)
    print('Saving binary mask numpy arrays')
    np.save(f'{binary_path}/binary_mask_of_tbms', binary_mask_of_tbms)
    np.save(f'{binary_path}/binary_mask_of_tubules', binary_mask_of_tubules)


def process_image(binaries_dir, wsi, filtered_wsi_dir):
    wsi_name = str(wsi.split('.')[0])
    if os.path.exists(os.path.join(filtered_wsi_dir, f'{wsi_name}.png')):
        path_ = os.path.join(filtered_wsi_dir, f'{wsi_name}.png')
        print(f'{path_} already exists')
        return
    binary_path = os.path.join(binaries_dir, wsi_name)
    binary_mask_of_tubules = np.load(f'{binary_path}/binary_mask_of_tubules.npy')
    binary_mask_of_tbms = np.load(f'{binary_path}/binary_mask_of_tbms.npy')
    print(f'performing distance transform - {wsi_name}')
    logging.info(f'performing distance transform - {wsi_name}')
    distance_transform = cv2.distanceTransform(binary_mask_of_tubules, cv2.DIST_L2, 5)
    inverse_binary_mask = 255 - binary_mask_of_tubules
    inverse_distance_transform = cv2.distanceTransform(inverse_binary_mask, cv2.DIST_L2, 5)
    distance_map = distance_transform - inverse_distance_transform
    
    del binary_mask_of_tubules  # Delete array after usage
    del inverse_binary_mask     # Delete array after usage
    del distance_transform      # Delete array after usage
    del inverse_distance_transform  # Delete array after usage
    gc.collect()
    
    threshold = 10
    mask = np.abs(distance_map) < threshold
    # Convert boolean mask to binary (0 and 255)
    threshInv = np.where(mask, 255, 0).astype(np.uint8)
    product = threshInv * binary_mask_of_tbms
    
    del distance_map  # Delete array after usage
    del mask          # Delete array after usage
    del threshInv 
    gc.collect()
    
    print(f'Saving filtered image - {wsi_name}')
    logging.info(f'Saving filtered image - {wsi_name}')
    filtered_image_path = os.path.join(filtered_wsi_dir, f'{wsi_name}.png')
    cv2.imwrite(filtered_image_path, 255 * product)
    
    del binary_mask_of_tbms  # Delete array after usage
    del product  
    
    gc.collect()
    print(f'Successfully filtered image - {wsi_name}')
    logging.info(f'Successfully filtered image - {wsi_name}')

    
def filter(tbm_config):
    try:
        WSI_images_dir = tbm_config.wsis_path
        tbm_annotations_dir = tbm_config.predicted_annotations_path
        tubules_annotations_dir = tbm_config.tubule_annotations
        binaries_dir = tbm_config.binaries
        makeDir(binaries_dir)

        WSI_images = os.listdir(WSI_images_dir)
        print(WSI_images)
        logging.info(f'Saving binary masks and numpy arrays for the WSIS in {WSI_images_dir}')
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(save_image_masks, wsi, WSI_images_dir, tbm_annotations_dir, tubules_annotations_dir, binaries_dir) for wsi in WSI_images]
            for future in futures:
                future.result()  # This will raise exceptions if any

        logging.info(f'Succesfully saved binary masks and numpy arrays for the WSIS in {WSI_images_dir}')


        filtered_wsi_dir = tbm_config.filtered_wsis
        makeDir(filtered_wsi_dir)

        logging.info(f'Started applying Post Processing on each WSI image')
        for wsi in WSI_images:
          process_image(binaries_dir, wsi, filtered_wsi_dir)  
        logging.info(f'Finished applying Post Processing on each WSI image')
    except Exception as e:
        print(f'Exception while Post processing: {str(e)}')
        logging.error(f'Exception while Post processing:{str(e)}')
        logging.error("Traceback: " + traceback.format_exc())



