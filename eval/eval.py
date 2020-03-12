#!/usr/bin/python
#coding=utf8

import numpy as np
import argparse
import gdal
from scipy.ndimage import sobel
from numpy.linalg import norm

import sys
import metrics as mtc

# import sewar.full_ref as sewar
# import sewar.no_ref as sewar_n

def q4(ms, ps):
    sigma1 = np.var(ms, axis=2) * 4 / 3
    sigma2 = np.var(ps, axis=2) * 4 / 3
    mu1 = np.mean(ms, axis=2)
    mu2 = np.mean(ps, axis=2)
  
    cov = np.zeros((128, 128), dtype=np.float64)
    for i in range(128):
        for j in range(128):
            for k in range(4):
                cov[i, j] += (ms[i, j, k] - mu1[i, j]) * (ps[i, j, k] - mu2[i, j])
            cov[i, j] /= 3 

    res = 4*np.abs(cov*mu1*mu2) / (sigma1+sigma2) / (mu1*mu1+mu2*mu2)
    is_nan = np.nonzero(np.isnan(res))
    for (x, y) in zip(is_nan[0], is_nan[1]):
        res[x, y] = 1

    return np.mean(res)


def sCC(ms, ps):
    ps_sobel = sobel(ps, mode='constant')
    ms_sobel = sobel(ms, mode='constant')

    return  np.mean(np.sum(ps_sobel*ms_sobel,axis=(0,1))/np.sqrt(np.sum(ps_sobel*ps_sobel,axis=(0,1)))/np.sqrt(np.sum(ms_sobel*ms_sobel,axis=(0,1))))

def sam(ms, ps):
    assert ms.ndim == 3 and ms.shape == ps.shape
    dot_sum = np.sum(ms * ps, axis=2)
    norm_true = norm(ms, axis=2)
    norm_pred = norm(ps, axis=2)

    res = np.arccos(dot_sum/norm_pred/norm_true)
    is_nan = np.nonzero(np.isnan(res))
    for (x,y) in zip(is_nan[0], is_nan[1]):
        res[x,y]=0

    return np.mean(res)

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="input directory")
parser.add_argument("--num", type=int, help="number of input image pairs")
parser.add_argument("--blk", type=int, help="size of sample lr image")
parser.add_argument("--satellite", help="satellite")

a = parser.parse_args()

for k, v in a._get_kwargs():
    print (k, "=", v)

Q4 = []
ERGAS = []
RASE = []
SCC = [] 
SAM = []
D_lambda = []
D_s = []
QNR = []
CC = []

for i in range(a.num):
    print ("remain: ", a.num - i)
    # test_0-outputs.tif
    x = '%s/images/%s_test_%d_%d-outputs.tif' % (a.input_dir, a.satellite, a.blk, i)
    y = '%s/images/%s_test_%d_%d-targets.tif' % (a.input_dir, a.satellite, a.blk, i)
    blur = '%s/images/%s_test_%d_%d-inputs1.tif' % (a.input_dir, a.satellite, a.blk, i)
    pan = '%s/images/%s_test_%d_%d-inputs2.tif' % (a.input_dir, a.satellite, a.blk, i)

    x = gdal.Open(x).ReadAsArray().transpose(1, 2, 0)
    y = gdal.Open(y).ReadAsArray().transpose(1, 2, 0)
    blur = gdal.Open(blur).ReadAsArray().transpose(1, 2, 0)
    pan = gdal.Open(pan).ReadAsArray()

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    blur = np.array(blur, dtype=np.float32)
    pan = np.array(pan, dtype=np.float32)

    ### ref 

    # SCC.append(sCC(y, x))
    # SAM.append(sam(y, x))
    # Q4.append(q4(y, x))
    
    ''' ERGAS.append(sewar.ergas(y, x))
    RASE.append(sewar.rase(y, x))
    SCC.append(sewar.scc(y, x))
    SAM.append(sewar.sam(y, x)) '''
    
    Q4.append(mtc.Q4(y, x))
    ERGAS.append(mtc.ERGAS(y, x))
    SCC.append(mtc.sCC(y, x))
    SAM.append(mtc.SAM(y, x))
    CC.append(mtc.CC(y, x))

    ### no_ref

    ''' D_lambda.append(sewar_n.d_lambda(blur, x))
    D_s.append(sewar_n.d_s(pan, blur, x))
    QNR.append(sewar_n.qnr(pan, blur, x)) '''

    D_lambda.append(mtc.D_lamda(x, blur))
    D_s.append(mtc.D_s(x, blur, pan))


SAM = np.array(SAM)
CC = np.array(CC)
SCC = np.array(SCC)
ERGAS = np.array(ERGAS)
Q4 = np.array(Q4)
D_lambda = np.array(D_lambda)
D_s = np.array(D_s)

print ("SAM: %.4lf+-%.4lf" % (np.mean(SAM) * 180 / np.pi, np.var(SAM * 180 / np.pi)))
print ("CC: %.4lf+-%4.lf" % (np.mean(CC), np.var(CC)))
print ("sCC: %.4lf+-%.4lf" % (np.mean(SCC), np.var(SCC)))
print ("ERGAS: %.4lf+-%.4lf " % (np.mean(ERGAS), np.var(ERGAS)))
print ("Q4: %.4lf+-%.4lf" % (np.mean(Q4), np.var(Q4)))
print ("D_lambda: %.4lf+-%.4lf" % (np.mean(D_lambda), np.var(D_lambda)))
print ("D_s: %.4lf+-%.4lf" % (np.mean(D_s), np.var(D_s)))

QNR = (1 - D_s) * (1 - D_lambda)
print ("QNR: %.4lf+-%.4lf" % (np.mean(QNR), np.var(QNR)))



