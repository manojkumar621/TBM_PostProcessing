#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
notebook_dir = os.getcwd()
project_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
if project_dir not in sys.path:
    sys.path.append(project_dir)
utils_dir = os.path.abspath(os.path.join(notebook_dir, '..', 'utils'))
if utils_dir not in sys.path:
    sys.path.append(utils_dir)
scripts_dir = os.path.abspath(os.path.join(notebook_dir, 'scripts'))
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)
from Split_WSI import split
from scripts.test import infer
from Save_annotations import save
from filter_masks_ import filter
from split_filtered_images import split_images
from hparams import HParam
from logging_setup import log_execution_time
import logging
import time

class TBMSegmentationPipeline:
    def __init__(self, tbm_config):
        self.tbm_config = tbm_config
    
    def split_wsi(self):        
        # Split WSI images logic
        wsis_path = self.tbm_config.wsis_path
        if os.path.exists(wsis_path):
            print(f'tiles path exists, hence not splitting any wsis')
            logging.info(f'tiles path exists, hence not splitting any wsis')
            return
        tiles_path = self.tbm_config.wsi_tiles_path
        split(wsis_path, tiles_path)

    def run_model_prediction(self):
        predicted_tiles_path = self.tbm_config.pred
        if os.path.exists(predicted_tiles_path):
            print(f'predicted tiles path exists, hence not running any predictions')
            logging.info(f'predicted tiles path exists, hence not running any predictions')
            return
        logging.info(f'started inference on tiles - {self.tbm_config.wsis_path}')
        infer(self.tbm_config)
        logging.info(f'Finished inference on tiles')

    def save_annotations(self, predicted = True):
        if predicted:
            annotations_path = self.tbm_config.predicted_annotations_path
        else:
            annotations_path = self.tbm_config.filtered_annotations_path
        if os.path.exists(annotations_path):
            print(f'annotations_path path exists, hence not saving any annotations')
            logging.info(f'annotations_path path exists, hence not saving any annotations')
            return
        # Save annotations logic
        logging.info(f'started Saving annotations to - {annotations_path}')
        save(self.tbm_config, predicted)
        logging.info(f'Finished Saving annotations')
        
    def filter_masks(self):
        filtered_path = self.tbm_config.filtered_wsis
        if os.path.exists(filtered_path):
            print(f'filtered tiles path exists, hence not filtering any WSIs')
            logging.info(f'filtered tiles path exists, hence not filtering any WSIs')
            return
        logging.info(f'started filtering tiles to - {filtered_path}')
        filter(self.tbm_config)
        logging.info(f'Finished saving filtered tiles')
    
    def split_filtered_images(self):
        filtered_path = self.tbm_config.filtered_tiles
        if os.path.exists(filtered_path):
            print(f'filtered tiles path exists, hence not splitting any WSIs')
            logging.info(f'filtered tiles path exists, hence not splitting any WSIs')
            return
        logging.info(f'started Splitting filtered WSIs to - {filtered_path}')
        split_images(self.tbm_config)
        logging.info(f'Finished Splitting filtered WSIs')

    def run(self):
        print('started running')
        self.split_wsi()
        self.run_model_prediction()
        self.save_annotations()
        self.filter_masks()
        self.split_filtered_images()
        self.save_annotations(predicted=False)
        print('success')
        
if __name__ == "__main__":
    tbm_config_path = "/home/manojkumargalla/PostProcess/config/tbm_config.yaml"
    tbm_config = HParam(tbm_config_path)
    pipeline = TBMSegmentationPipeline(tbm_config)
    pipeline.run()


# In[ ]:




