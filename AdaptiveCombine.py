import numpy as np
#from matplotlib import pyplot as plt
from numpy import resize
from scipy import io as sio
import scipy.linalg as la
import torch
# import sigpy.plot as pl
from torch.fft import ifft, fft
import time
from multiprocessing import Process
import os
from multiprocessing import Pool
import cv2
import sigpy as sp

def AdaptiveCombine(**kwargs):
    im = kwargs['im']
    [nc_tmp, ny_tmp, nx_tmp] = np.shape(im)

    num_pad = 4 

    im = sp.resize(im, (nc_tmp,ny_tmp+num_pad,nx_tmp+num_pad))

    [nc, ny, nx] = np.shape(im)

    coil_dim_sum = np.sum(np.sum(np.transpose(abs(im),(2,1,0)),axis = 0),axis = 0)
    mm = max(coil_dim_sum)
    maxcoil = np.where(coil_dim_sum == mm)


    if len(kwargs) < 3:
        rn = np.eye(nc)  

    if len(kwargs) < 2:
        donorm = 0

    bs1 = 4  
    bs2 = 4  
    st = 2  

    wsmall = np.zeros((nc, round(ny//st), nx//st), dtype=complex)
    cmapsmall = np.zeros((nc, round(ny // st), nx // st), dtype=complex)

    for x in range(st, nx, st):
        for y in range(st, ny, st):
            ymin1 = max([y - bs1//2, 0])
            xmin1 = max([x - bs2//2, 0])

            ymax1 = min([y + bs1//2, ny-1])
            xmax1 = min([x + bs2//2, nx-1])

            ly1 = ymax1 - ymin1 + 1
            lx1 = xmax1 - xmin1 + 1

            im1 = im[:, ymin1:ymax1+1, xmin1:xmax1+1]

            m1 = im1.reshape((nc, lx1 * ly1))
            m = np.dot(m1, np.conj(np.transpose(m1)))

            v, e = la.eig(np.dot(np.linalg.inv(rn), m)) 
            mv = max(v)  
            ind = np.where(v == mv)

            mf = np.squeeze(e[:,ind],2) 
            mf = mf / (np.dot(np.dot(np.conj(np.transpose(mf)),np.linalg.inv(rn)),mf))
            normmf = np.squeeze(e[:,ind],2)

            mf = mf*np.exp(-1j * np.angle(mf[maxcoil]))
            normmf = normmf*np.exp(-1j * np.angle(normmf[maxcoil]))

            wsmall[:, y//st, x//st] = np.squeeze(mf)
            cmapsmall[:, y//st, x//st] = np.squeeze(normmf)

    recon = np.zeros((ny, nx))

    wfull = np.zeros((nc, ny, nx), dtype=complex)
    cmap = np.zeros((nc, ny, nx), dtype=complex)

    for i in range(nc):
        real_wsmall = wsmall[i, :, :].real
        real_cmapsmall = cmapsmall[i, :, :].real
        imag_wsmall = wsmall[i, :, :].imag
        imag_cmapsmall = cmapsmall[i, :, :].imag

        real_wsmall = cv2.resize(real_wsmall, (nx, ny), interpolation=cv2.INTER_NEAREST)
        real_cmapsmall = cv2.resize(real_cmapsmall, (nx, ny), interpolation=cv2.INTER_NEAREST)
        imag_wsmall = cv2.resize(imag_wsmall, (nx, ny), interpolation=cv2.INTER_NEAREST)
        imag_cmapsmall = cv2.resize(imag_cmapsmall, (nx, ny), interpolation=cv2.INTER_NEAREST)

        wfull[i, :, :] = np.conj(real_wsmall + 1j*imag_wsmall)
        cmap[i, :, :] = real_cmapsmall + 1j*imag_cmapsmall

    for i in range(nc):
        recon = recon + np.squeeze(wfull[i, :, :]*im[i, :, :])

    if donorm:
        recon = recon * np.squeeze(np.sum(abs(cmap)**2, axis=0))

    recon = sp.resize(recon,(ny_tmp, nx_tmp))
    wfull = sp.resize(wfull,(nc_tmp, ny_tmp, nx_tmp))
    return recon, wfull