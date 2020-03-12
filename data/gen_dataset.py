############## import ##############
from __future__ import division
import cv2
import gdal, ogr, os, osr
import numpy as np
import random

import sys
sys.path.append('../')
from utils import array2raster

############## arguments ##############
dataDir = '/data/zh/PSData' # root of data directory 
satellite = 'WV-2'          # name of satellite
tot = 9                     # number of images 
blk = 64                    # patch size
ratio = 10                  # control the number of training dataset
trainIndex = range(2, tot + 1) # images index for training set
testIndex = range(1, 2)        # images index for testing set
originTestIndex = range(1, 2)  # images index for origin_test set

if __name__ == "__main__":
    rasterOrigin = (0, 0)
    
    trainCount = 0
    testCount = 0
    originTestCount = 0

    trainDir = "%s/Dataset/%s/train_%d" % (dataDir, satellite, blk)
    testDir = "%s/Dataset/%s/test_%d" % (dataDir, satellite, blk)
    originTestDir = "%s/Dataset/%s/test_origin_%d" % (dataDir, satellite, blk)

    ### train 
    if not os.path.exists(trainDir):
        os.makedirs(trainDir)
    record = open('%s/record.txt' % trainDir, "w")
    for num in trainIndex:
        mul = '%s/Dataset/%s/%d_mul.tif' % (dataDir, satellite, num)
        lr = '%s/Dataset/%s/%d_lr.tif' % (dataDir, satellite, num)
        lr_u = '%s/Dataset/%s/%d_lr_u.tif' % (dataDir, satellite, num)
        pan = '%s/Dataset/%s/%d_pan.tif' % (dataDir, satellite, num)
        pan_d = '%s/Dataset/%s/%d_pan_d.tif' % (dataDir, satellite, num)

        dt_mul = gdal.Open(mul)
        dt_lr = gdal.Open(lr)
        dt_lr_u = gdal.Open(lr_u)
        dt_pan = gdal.Open(pan)
        dt_pan_d = gdal.Open(pan_d)
        
        img_mul = dt_mul.ReadAsArray() # (c, h, w)
        img_lr = dt_lr.ReadAsArray()
        img_lr_u = dt_lr_u.ReadAsArray()
        img_pan = dt_pan.ReadAsArray()
        img_pan_d = dt_pan_d.ReadAsArray()    

        XSize = dt_lr.RasterXSize
        YSize = dt_lr.RasterYSize
        
        sample = int (XSize * YSize / blk / blk * ratio)

        for _ in range(sample):
            x = random.randint(0, XSize - blk)
            y = random.randint(0, YSize - blk)

            array2raster('%s/%d_mul.tif' % (trainDir, trainCount), rasterOrigin, 2.4, 2.4,
                img_mul[:, y * 4:(y + blk) * 4, x * 4:(x + blk) * 4], 4)
            array2raster('%s/%d_lr_u.tif' % (trainDir, trainCount), rasterOrigin, 2.4, 2.4,
                img_lr_u[:, y * 4:(y + blk) * 4, x * 4:(x + blk) * 4], 4)
            array2raster('%s/%d_lr.tif' % (trainDir, trainCount), rasterOrigin, 2.4, 2.4,
                img_lr[:, y:(y + blk), x:(x + blk)], 4)
            array2raster('%s/%d_pan.tif' % (trainDir, trainCount), rasterOrigin, 2.4, 2.4,
                img_pan[y * 4:(y + blk) * 4, x * 4:(x + blk) * 4], 1)
            array2raster('%s/%d_pan_d.tif' % (trainDir, trainCount), rasterOrigin, 2.4, 2.4,
                img_pan_d[y * 4:(y + blk) * 4, x * 4:(x + blk) * 4], 1)    
            trainCount += 1

        print ("done %d" % num)

    record.write("%d\n" % trainCount)
    record.close()  

    ### test 
    if not os.path.exists(testDir):
        os.makedirs(testDir)
    record = open('%s/record.txt' % testDir, "w")
    for num in testIndex:
        mul = '%s/Dataset/%s/%s_mul.tif' % (dataDir, satellite, num)
        lr = '%s/Dataset/%s/%s_lr.tif' % (dataDir, satellite, num)
        lr_u = '%s/Dataset/%s/%s_lr_u.tif' % (dataDir, satellite, num)
        pan = '%s/Dataset/%s/%s_pan.tif' % (dataDir, satellite, num)
        pan_d = '%s/Dataset/%s/%s_pan_d.tif' % (dataDir, satellite, num)

        dt_mul = gdal.Open(mul)
        dt_lr = gdal.Open(lr)
        dt_pan = gdal.Open(pan)
        dt_pan_d = gdal.Open(pan_d)
        dt_lr_u = gdal.Open(lr_u)
        
        img_mul = dt_mul.ReadAsArray()
        img_lr = dt_lr.ReadAsArray()
        img_pan = dt_pan.ReadAsArray()
        img_pan_d = dt_pan_d.ReadAsArray()
        img_lr_u = dt_lr_u.ReadAsArray()

        XSize = dt_lr.RasterXSize
        YSize = dt_lr.RasterYSize
        
        row = 0
        col = 0
        
        for y in range(0, YSize, blk // 8 * 7): # 按顺序切(32, 32)小块
            if y + blk > YSize:
                continue
            col = 0
            
            for x in range(0, XSize, blk // 8 * 7):
                if x + blk > XSize:
                    continue

                array2raster('%s/%d_mul.tif' % (testDir, testCount), rasterOrigin, 2.4, 2.4,
                     img_mul[:, y * 4:(y + blk) * 4, x * 4:(x + blk) * 4], 4)
                array2raster('%s/%d_lr_u.tif' % (testDir, testCount), rasterOrigin, 2.4, 2.4,
                     img_lr_u[:, y * 4:(y + blk) * 4, x * 4:(x + blk) * 4], 4)
                array2raster('%s/%d_lr.tif' % (testDir, testCount), rasterOrigin, 2.4, 2.4,
                     img_lr[:, y:(y + blk), x:(x + blk)], 4)
                array2raster('%s/%d_pan.tif' % (testDir, testCount), rasterOrigin, 2.4, 2.4,
                     img_pan[y * 4:(y + blk) * 4, x * 4:(x + blk) * 4], 1)
                array2raster('%s/%d_pan_d.tif' % (testDir, testCount), rasterOrigin, 2.4, 2.4,
                     img_pan_d[y * 4:(y + blk) * 4, x * 4:(x + blk) * 4], 1)
                
                testCount += 1      
                col += 1
            
            row += 1
            print (num, row)
        
        record.write("%d: %d * %d\n" % (num, row, col))
        
    record.write("%d\n" % testCount)
    record.close()
    print ("done") 

    ### origin test 
    if not os.path.exists(originTestDir):
        os.makedirs(originTestDir)
    record = open('%s/record.txt' % originTestDir, "w")
    for num in originTestIndex:
        lr = '%s/Dataset/%s/%s_mul_o.tif' % (dataDir, satellite, num)
        lr_u = '%s/Dataset/%s/%s_mul_o_u.tif' % (dataDir, satellite, num)
        pan = '%s/Dataset/%s/%s_pan_o.tif' % (dataDir, satellite, num)
        pan_d = '%s/Dataset/%s/%s_pan_o_d.tif' % (dataDir, satellite, num)

        dt_lr = gdal.Open(lr)
        dt_pan = gdal.Open(pan)
        dt_pan_d = gdal.Open(pan_d)
        dt_lr_u = gdal.Open(lr_u)
        
        img_lr = dt_lr.ReadAsArray()
        img_pan = dt_pan.ReadAsArray()
        img_pan_d = dt_pan_d.ReadAsArray()
        img_lr_u = dt_lr_u.ReadAsArray()

        XSize = dt_lr.RasterXSize
        YSize = dt_lr.RasterYSize
        
        row = 0
        col = 0
        
        for y in range(0, YSize, blk // 8 * 7): # 按顺序切(32, 32)小块
            if y + blk > YSize:
                continue
            col = 0
            
            for x in range(0, XSize, blk // 8 * 7):
                if x + blk > XSize:
                    continue

                array2raster('%s/%d_lr_u.tif' % (originTestDir, originTestCount), rasterOrigin, 2.4, 2.4,
                     img_lr_u[:, y * 4:(y + blk) * 4, x * 4:(x + blk) * 4], 4)
                array2raster('%s/%d_lr.tif' % (originTestDir, originTestCount), rasterOrigin, 2.4, 2.4,
                     img_lr[:, y:(y + blk), x:(x + blk)], 4)
                array2raster('%s/%d_pan.tif' % (originTestDir, originTestCount), rasterOrigin, 2.4, 2.4,
                     img_pan[y * 4:(y + blk) * 4, x * 4:(x + blk) * 4], 1)
                array2raster('%s/%d_pan_d.tif' % (originTestDir, originTestCount), rasterOrigin, 2.4, 2.4,
                     img_pan_d[y * 4:(y + blk) * 4, x * 4:(x + blk) * 4], 1)
                
                originTestCount += 1      
                col += 1
            
            row += 1
            print (num, row)
        
        record.write("%d: %d * %d\n" % (num, row, col))
        
    record.write("%d\n" % originTestCount)
    record.close()
    print ("done") 
