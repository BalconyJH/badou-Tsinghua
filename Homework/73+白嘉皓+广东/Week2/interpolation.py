#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Time:4/01/2021 5:12 pm
# Author:BalconyJH
# Site:
# File:interpolation.py
# Version:1.1


# bilinear interpolation
import cv2
import numpy as np

def func(img, out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


if __name__ == '__main__':
    img = cv2.imread('thumbsup.jpg')
    dst = func(img, (1024, 1024))
    cv2.imshow('bilinear interp', dst)
    cv2.imwrite("1.png", dst)
    cv2.waitKey()

'''
#NN interpolation
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 
import math

def NN_interpolation(img,dstH,dstW):
    scrH,scrW,_=img.shape
    retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
    for i in range(dstH-1):
        for j in range(dstW-1):
            scrx=round(i*(scrH/dstH))
            scry=round(j*(scrW/dstW))
            retimg[i,j]=img[scrx,scry]
    return retimg

im_path='../paojie.jpg'
image=np.array(Image.open(im_path))

image1=NN_interpolation(image,image.shape[0]*2,image.shape[1]*2)
image1=Image.fromarray(image1.astype('uint8')).convert('RGB')
image1.save('out.png')
'''