#!/usr/bin/python
#coding=utf8

import numpy as np
import argparse
import gdal
from scipy.ndimage import sobel
from numpy.linalg import norm

import sys
import metrics as mtc

satellites = ["QB", "GF-2"]
methods = ["BDSD", "GS", "IHS", "HCS", "Brovey", "HPFC", "HPF", "LMM", "LMVM", "SFIM"]
inputDir = '/data/zh/PSOutput/PanFusion_origin_result'
for satellite in satellites:
    for method in methods:
        print (satellite, method)

        x = '%s/%s_%s_test_lr_u.tif' % (inputDir, method, satellite)
        # y = '/data/zh/PSOutput/PanFusion_result/%s_test_mul.tif' % satellite
        blur = '%s/%s_test_lr.tif' % (inputDir, satellite)
        pan = '%s/%s_test_pan.tif' % (inputDir, satellite)

        x = gdal.Open(x).ReadAsArray().transpose(1, 2, 0)
        # y = gdal.Open(y).ReadAsArray().transpose(1, 2, 0)
        blur = gdal.Open(blur).ReadAsArray().transpose(1, 2, 0)
        pan = gdal.Open(pan).ReadAsArray()

        x = np.array(x, dtype=np.float32)
        # y = np.array(y, dtype=np.float32)
        blur = np.array(blur, dtype=np.float32)
        pan = np.array(pan, dtype=np.float32)

        """ SAM = mtc.SAM(y, x) * 180 / np.pi
        CC = mtc.CC(y, x)
        SCC = mtc.sCC(y, x)
        ERGAS = mtc.ERGAS(y, x)
        Q4 = mtc.Q4(y, x) """
        D_lambda = mtc.D_lamda(x, blur)
        D_s = mtc.D_s(x, blur, pan)
        QNR = (1 - D_s) * (1 - D_lambda)

        """ print("SAM: %.4lf" % SAM)
        print("CC: %.4lf" % CC)
        print("sCC: %.4lf" % SCC)
        print("ERGAS: %.4lf" % ERGAS)
        print("Q4: %.4lf" % Q4) """
        print("D_lambda: %.4lf" % D_lambda)
        print("D_s: %.4lf" % D_s)
        print("QNR: %.4lf" % QNR)


