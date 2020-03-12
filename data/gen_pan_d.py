from __future__ import division
import cv2
import gdal, ogr, os, osr
import numpy as np
import random

import sys
sys.path.append('../')
from utils import array2raster

if __name__ == "__main__":
    satellite = 'GF-2'
    mode = 'test'
    blk = 64
    
    train_cnt ={ 32: {"QB": 53857, "GF-2": 83062, "QB_origin": 0, "GF-2_origin": 0},
                 64: {"QB": 13460, "GF-2": 20765, "QB_origin": 0, "GF-2_origin": 0} }
    test_cnt = { 32: {"QB": 486, "GF-2": 3965, "QB_origin": 486, "GF-2_origin": 3965},
                 64: {"QB": 117, "GF-2": 960, "QB_origin": 117, "GF-2_origin": 960} }
    
    cnt = train_cnt[blk][satellite] if mode == 'train' else test_cnt[blk][satellite]
    dataDir = '/data/zh/PSData/Dataset/%s/%s_%d' % (satellite, mode, blk)
    rasterOrigin = (-123.25745,45.43013)

    for num in range(cnt):
        dt_pan = gdal.Open('%s/%d_pan.tif' % (dataDir, num)).ReadAsArray()
        dt_pan_d = cv2.resize(cv2.resize(dt_pan, (blk, blk)), (blk * 4, blk * 4))
        
        array2raster('%s/%d_pan_d.tif' % (dataDir, num), rasterOrigin, 2.4, 2.4, dt_pan_d, 1)
    
    print ("done")