############## import ##############
from __future__ import division
import gdal, ogr, os, osr
import numpy as np
import cv2

import sys
sys.path.append('../')
from utils import array2raster

############## arguments ##############
dataDir = '/data/zh/PSData' # root of data directory
satellite = 'WV-2'          # name of dataset
tot = 9                     # number of raw images
imagesIndex1 = range(1, tot + 1) # image index for training and testing set
imagesIndex2 = range(1, 2)       # image index for origin_test set

def downsample(img, ratio=4):
    h, w = img.shape[:2]
    return cv2.resize(img, (w // ratio, h // ratio))
def upsample(img, ratio=4):
    h, w = img.shape[:2]
    return cv2.resize(img, (w * ratio, h * ratio))

if __name__ == "__main__":
    rasterOrigin = (0, 0)

    outDir = '%s/Dataset/%s' % (dataDir, satellite)
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    # MUL -> mul(1), lr(1/4), lr_u(1/4*4)
    # PAN -> pan(1/4), pan_d(1/4*1/4*4)
    for i in imagesIndex1: 
        newMul = '%s/Dataset/%s/%d_mul.tif' % (dataDir, satellite , i)
        newLR = '%s/Dataset/%s/%d_lr.tif' % (dataDir, satellite, i)
        newLR_U = '%s/Dataset/%s/%d_lr_u.tif' % (dataDir, satellite, i)
        newPan = '%s/Dataset/%s/%d_pan.tif' % (dataDir, satellite, i)
        newPan_D = '%s/Dataset/%s/%d_pan_d.tif' % (dataDir, satellite, i)

        rawMul = gdal.Open( '%s/Raw/%s/%d-MUL.TIF' % (dataDir, satellite, i) ).ReadAsArray()
        rawPan = gdal.Open( '%s/Raw/%s/%d-PAN.TIF' % (dataDir, satellite, i) ).ReadAsArray()
        print ("rawMul:", rawMul.shape, " rawPan:", rawPan.shape)

        rawMul = rawMul.transpose(1, 2, 0) # (h, w, c)
        
        h, w = rawMul.shape[:2]
        h = h // 4 * 4
        w = w // 4 * 4
        
        imgMul = cv2.resize(rawMul, (w, h))
        imgLR = cv2.resize(imgMul, (w // 4, h // 4))
        imgLR_U = cv2.resize(imgLR, (w, h))
        imgPan = cv2.resize(rawPan, (w, h))
        imgPan_D = upsample(downsample(imgPan))

        imgMul = imgMul.transpose(2, 0, 1)
        imgLR = imgLR.transpose(2, 0, 1)
        imgLR_U = imgLR_U.transpose(2, 0, 1)
        
        array2raster(newMul, rasterOrigin, 2.4, 2.4, imgMul, 4)        # mul
        array2raster(newLR_U, rasterOrigin, 2.4, 2.4, imgLR_U, 4)     # lr_u
        array2raster(newLR, rasterOrigin, 2.4, 2.4, imgLR, 4)        # lr
        array2raster(newPan, rasterOrigin, 2.4, 2.4, imgPan, 1)       # pan
        array2raster(newPan_D, rasterOrigin, 2.4, 2.4, imgPan_D, 1)       # pan_d
        
        print ('mul:', imgMul.shape, ' lr_u:', imgLR_U.shape, 
            ' lr:', imgLR.shape, ' pan:', imgPan.shape, ' pan_d:', imgPan_D.shape)
        print ('done%s' % i)

    # origin 
    # MUL(crop 1/4) -> mul_o(1), mul_o_u(1*4)
    # PAN(crop 1/4) -> pan_o(1), pan_o_d(1/4*4)
    for i in imagesIndex2: 
        newMul_o = '%s/Dataset/%s/%d_mul_o.tif' % (dataDir, satellite , i)
        newMul_o_u = '%s/Dataset/%s/%d_mul_o_u.tif' % (dataDir, satellite, i)
        newPan_o = '%s/Dataset/%s/%d_pan_o.tif' % (dataDir, satellite, i)
        newPan_o_d = '%s/Dataset/%s/%d_pan_o_d.tif' % (dataDir, satellite, i)

        rawMul = gdal.Open( '%s/Raw/%s/%d-MUL.TIF' % (dataDir, satellite, i) ).ReadAsArray()
        rawPan = gdal.Open( '%s/Raw/%s/%d-PAN.TIF' % (dataDir, satellite, i) ).ReadAsArray()
        print ("rawMul:", rawMul.shape, " rawPan:", rawPan.shape)

        rawMul = rawMul.transpose(1, 2, 0) # (h, w, c)

        h, w = rawMul.shape[:2]
        h = h // 4 * 4
        w = w // 4 * 4

        imgMul_o = cv2.resize(rawMul, (w, h))[1500:1500+h//4, 1500:1500+w//4, :] # (1/4, 1/4)
        imgPan_o = cv2.resize(rawPan, (w*4, h*4))[6000:6000+h, 6000:6000+w]      # (1, 1)
        imgMul_o_u = upsample(imgMul_o)
        imgPan_o_d = upsample(downsample(imgPan_o))

        imgMul_o = imgMul_o.transpose(2, 0, 1)
        imgMul_o_u = imgMul_o_u.transpose(2, 0, 1)

        array2raster(newMul_o, rasterOrigin, 2.4, 2.4, imgMul_o, 4)        
        array2raster(newMul_o_u, rasterOrigin, 2.4, 2.4, imgMul_o_u, 4)     
        array2raster(newPan_o, rasterOrigin, 2.4, 2.4, imgPan_o, 1)        
        array2raster(newPan_o_d, rasterOrigin, 2.4, 2.4, imgPan_o_d, 1)      

        print ('mul_o:', imgMul_o.shape, ' mul_o_u:', imgMul_o_u.shape, 
            ' pan_o:', imgPan_o.shape, ' pan_o_d:', imgPan_o_d.shape, )
        print ('done %d' % i) 
