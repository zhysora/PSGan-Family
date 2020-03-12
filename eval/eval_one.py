#!/usr/bin/python
#coding=utf8

############## import ##############
import numpy as np
import argparse
import gdal
from scipy.ndimage import sobel
from numpy.linalg import norm
import os

import sys
import metrics as mtc
sys.path.append("../")
from utils import array2raster

############## arguments ##############
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="input directory")
parser.add_argument("--num", type=int, help="number of input image pairs")
parser.add_argument("--blk", type=int, help="size of sample lr image")
parser.add_argument("--col", type=int, help="col")
parser.add_argument("--row", type=int, help="row")
parser.add_argument("--ref", type=int, help="reference metrics")

a = parser.parse_args()

for k, v in a._get_kwargs():
    print (k, "=", v)

if __name__ == "__main__":
    col_sz = a.col
    row_sz = a.row

    YSize = (a.blk // 8 * 7 * (a.row - 1) + a.blk) * 4
    XSize = (a.blk // 8 * 7 * (a.col - 1) + a.blk) * 4

    mul = np.zeros(shape=[YSize, XSize, 4], dtype=np.float32)
    pan = np.zeros(shape=[YSize, XSize], dtype=np.float32)
    # pan_d = np.zeros(shape=[YSize, XSize], dtype=np.float32)
    blur = np.zeros(shape=[YSize//4, XSize//4, 4], dtype=np.float32)
    out = np.zeros(shape=mul.shape, dtype=np.float32)
    blur_u = np.zeros(shape=mul.shape, dtype=np.float32)
    cnt = np.zeros(shape=mul.shape, dtype=np.float32)
    cnt_ = np.zeros(shape=blur.shape, dtype=np.float32)

    print(mul.shape)

    i = 0
    y = 0

    dataDir = '%s/images/' % a.input_dir
    if a.ref == 0:
        dataDir = dataDir + 'origin_'

    for _ in range(a.row):
        x = 0
        for __ in range(a.col):
            img = '%stest_%d-blur.tif' % (dataDir, i)
            img = gdal.Open(img).ReadAsArray().transpose(1, 2, 0)
            img = np.array(img, dtype=np.float32)
            
            ly = y
            ry = (y + a.blk)
            lx = x
            rx = (x + a.blk)

            blur[ly:ry, lx:rx, :] = blur[ly:ry, lx:rx, :] + img
            cnt_[ly:ry, lx:rx, :] = cnt_[ly:ry, lx:rx, :] + 1

            
            img = '%stest_%d-pan.tif' % (dataDir, i)
            img = gdal.Open(img).ReadAsArray()
            img = np.array(img, dtype=np.float32)
            
            ly = y * 4
            ry = (y + a.blk) * 4
            lx = x * 4
            rx = (x + a.blk) * 4

            pan[ly:ry, lx:rx] = pan[ly:ry, lx:rx] + img
            cnt[ly:ry, lx:rx, :] = cnt[ly:ry, lx:rx, :] + 1

            if a.ref != 0:
                img = '%stest_%d-mul.tif' % (dataDir, i)
                img = gdal.Open(img).ReadAsArray().transpose(1, 2, 0)
                img = np.array(img, dtype=np.float32)
                
                mul[ly:ry, lx:rx, :] = mul[ly:ry, lx:rx, :] + img
            
            img = '%stest_%d-mul_hat.tif' % (dataDir, i)
            img = gdal.Open(img).ReadAsArray().transpose(1, 2, 0)
            img = np.array(img, dtype=np.float32)
            
            out[ly:ry, lx:rx, :] = out[ly:ry, lx:rx, :] + img

            img = '%stest_%d-blur_u.tif' % (dataDir, i)
            img = gdal.Open(img).ReadAsArray().transpose(1, 2, 0)
            img = np.array(img, dtype=np.float32)

            blur_u[ly:ry, lx:rx, :] = blur_u[ly:ry, lx:rx, :] + img
            
            i = i + 1
            x = x + a.blk // 8 * 7
        y = y + a.blk // 8 * 7

    blur = blur / cnt_
    mul = mul / cnt
    pan = pan / cnt[:, :, 0]
    out = out / cnt
    blur_u = blur_u / cnt

    ### you can use this code to save the evaluation images
    """ if not os.path.exists('out'):
            os.makedirs('out')
    array2raster("out/wv2_lr.tif", (0, 0), 1, 1, blur.transpose(2, 0, 1), 4)
    array2raster("out/wv2_lr_u.tif", (0, 0), 1, 1, blur_u.transpose(2, 0, 1), 4)
    array2raster("out/wv2_mul.tif", (0, 0), 1, 1, mul.transpose(2, 0, 1), 4)
    array2raster("out/wv2_pan_lr.tif", (0, 0), 1, 1, pan, 1)  
    array2raster("out/wv2_ratio.tif", (0, 0), 1, 1, out.transpose(2, 0, 1), 4) """ 

    if a.ref == 1:
        Q4 = []
        ERGAS = []
        RASE = []
        SCC = [] 
        SAM = []
        CC = []
        Q4.append(mtc.Q4(mul, out))
        ERGAS.append(mtc.ERGAS(mul, out))
        SCC.append(mtc.sCC(mul, out))
        SAM.append(mtc.SAM(mul, out))
        CC.append(mtc.CC(mul, out)) 
        SAM = np.array(SAM)
        CC = np.array(CC)
        SCC = np.array(SCC)
        ERGAS = np.array(ERGAS)
        Q4 = np.array(Q4) 
        print ("SAM: %.4lf" % (np.mean(SAM) * 180 / np.pi))
        print ("CC: %.4lf" % (np.mean(CC)))
        print ("sCC: %.4lf" % (np.mean(SCC)))
        print ("ERGAS: %.4lf" % (np.mean(ERGAS)))
        print ("Q4: %.4lf" % (np.mean(Q4))) 
    elif a.ref == 0:
        D_lambda = []
        D_s = []
        QNR = []
        D_lambda.append(mtc.D_lamda(out, blur))
        D_s.append(mtc.D_s(out, blur, pan))
        D_lambda = np.array(D_lambda)
        D_s = np.array(D_s)
        print ("D_lambda: %.4lf" % (np.mean(D_lambda)))
        print ("D_s: %.4lf" % (np.mean(D_s)))
        QNR = (1 - D_s) * (1 - D_lambda)
        print ("QNR: %.4lf" % (np.mean(QNR)))   

    print("done")