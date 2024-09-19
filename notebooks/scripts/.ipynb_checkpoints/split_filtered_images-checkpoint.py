from PIL import Image, ImageFilter
import os
import glob
import numpy as np
Image.MAX_IMAGE_PIXELS = None
from logging_setup import log_execution_time
import logging
import traceback
def makeDir(mydir):
    if not os.path.exists(mydir):
        os.mkdir(mydir)

def split_images(tbm_config):
    tileSize = 512
    outdirbase = tbm_config.filtered_tiles
    makeDir(outdirbase)

    # # Read large filtered PNG images
    datadir = tbm_config.filtered_wsis
    files = glob.glob(f"{datadir}/*.png")
    for fid in range(len(files)):
        filename = files[fid].split('/')[-1].replace('.png', '')
        try:
            image = Image.open(files[fid])
            nx, ny = image.size

            print(f'Processing file name: {filename} nx: {nx}    ny: {ny}')

            # Extract Binary
            # threshold = 200
            # binary = image.convert('L').point(lambda p: 255 if p < threshold else 0)
            binary = image.convert('L')
            binarynb = np.array(binary)
            # print('converted image')
            # binary = binary.filter(ImageFilter.MedianFilter(size=29))
            # print('fitered image')
            # binarynb = np.array(binary)

            # binarynb = np.load('/blue/pinaki.sarder/manojkumargalla/npy_files/product_thresh10.npy')

            outdirSlide = f"{outdirbase}/{filename}"
            makeDir(outdirSlide)
            print('Starting for loop')
            for ix in range(nx // tileSize):
                if ix % 10 == 0:
                    print(f"ix: {ix}/{nx // tileSize}")
                for iy in range(ny // tileSize):
                    xxx = int(ix * tileSize)
                    yyy = int(iy * tileSize)
                    slice_contains_255 = np.any(binarynb[yyy:yyy+tileSize, xxx:xxx+tileSize] == 255)
                    if slice_contains_255:
                        x = tileSize * ix
                        y = tileSize * iy

                        tile = image.crop((x, y, x + tileSize, y + tileSize))
                        # print(f'saving tile {outdirSlide}/{str(nx)}_{str(ny)}_{str(x).zfill(5)}x_{str(y).zfill(5)}y.png')
                        tile.save(f"{outdirSlide}/{str(nx)}_{str(ny)}_{str(x).zfill(5)}x_{str(y).zfill(5)}y.png")
        except Exception as e:
            print(f'Exception while Splitting Filtered Images: {str(e)}')
            logging.error(f'Exception while Splitting Filtered Images:{str(e)}')
            logging.error("Traceback: " + traceback.format_exc())

