

'''This cell uses python's multiprocessing library to parallelly process the splitting of images and uses cupy which is a Numpy-like library to enable GPU acceleration for image processing tasks'''
import cupy as cp
import multiprocessing as mp
from PIL import Image, ImageFilter
import tiffslide
import os
import glob
import numpy as np
from logging_setup import log_execution_time
import logging
import traceback
from functools import partial

    
def makeDir(mydir):
    if not os.path.exists(mydir):
        os.mkdir(mydir)

def process_file(file, tiles_path):
    # print(file)
    logging.info(file)
    filename = file.split('/')[-1].replace('.svs', '')
    try:
        tileSize=512
        slide = tiffslide.open_slide(file)
        nx, ny = slide.level_dimensions[0]
        ds = int(slide.level_downsamples[slide.level_count-1])
        logging.info(f'Splitting file name: {filename} nx: {nx}    ny: {ny}       downsample: {ds}')
        threshold = 200
        region = slide.read_region((0, 0), slide.level_count-1, slide.level_dimensions[slide.level_count-1])
        binary = region.convert('L').point(lambda p: 255 if p < threshold else 0)
        binary = binary.filter(ImageFilter.MedianFilter(size=29))
        binarynb = cp.array(binary)

        outdirSlide = f"{tiles_path}/{filename}"
        makeDir(outdirSlide)
        c=0
        for ix in range(nx // tileSize):
            if ix % 10 == 0:
                logging.info(f"ix: {ix}/{nx // tileSize}, saved {c} images!")
                c=0
            for iy in range(ny // tileSize):
                xxx = int((ix * tileSize) / ds)
                yyy = int((iy * tileSize) / ds)
                slice_contains_255 = np.any(binarynb[yyy:yyy+tileSize//ds, xxx:xxx+tileSize//ds] == 255)
                # x = tileSize * ix
                # y = tileSize * iy
                # slice_region = binarynb[y:y + tileSize, y:y + tileSize]
                if slice_contains_255:
                # if cp.any(slice_region == 255):
                    # print(f'This has ones!!! {filename}')
                    # print(f'xxx for {filename} = {xxx} & yyy for {filename} = {yyy}')
                    # return
                    c+=1
                    x = tileSize * ix
                    y = tileSize * iy
                    tile = slide.read_region((x, y), 0, (tileSize, tileSize))
                    tile.save(f"{outdirSlide}/{str(nx)}_{str(ny)}_{str(x).zfill(5)}x_{str(y).zfill(5)}y.png")
    except Exception as e:
        print(f'Exception while processing file name: {filename} - {str(e)}')
        logging.error(f'Exception while processing file name: {filename} - {str(e)}')
        logging.error("Traceback: " + traceback.format_exc())

# print('hello from split wsi!)'
# tileSize = 512
# outdirbase = f"/blue/pinaki.sarder/manojkumargalla/PostProcess/data/model2/batch1/3/tiles_3/"
# makeDir(outdirbase)
# datadir = "/blue/pinaki.sarder/manojkumargalla/PostProcess/data/model2/batch1/3/wsis_3/"
# files = glob.glob(f"{datadir}/*.svs")

# # Create a pool of workers and process files in parallel
# with mp.Pool(processes=mp.cpu_count()) as pool:
#     pool.map(process_file, files)
@log_execution_time
def split(wsis_path, tiles_path):
    try:
        logging.info(f'Began splitting tiles from {wsis_path}')
        makeDir(tiles_path)
        files = glob.glob(f"{wsis_path}/*.svs")
        process_file_with_params = partial(process_file, tiles_path=tiles_path)
        with mp.Pool(processes=mp.cpu_count()) as pool:
            pool.map(process_file_with_params, files)
        logging.info(f'Finished splitting tiles from {wsis_path}')
    except Exception as e:
        logging.error(f'There was an error in splitting WSI images into tiles')
        logging.error(e)
        logging.error("Traceback: " + traceback.format_exc())
