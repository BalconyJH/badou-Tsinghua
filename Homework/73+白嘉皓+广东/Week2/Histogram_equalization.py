#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Time:31/12/2020 2:17 pm
# Author:BalconyJH
# Site:
# File:Histogram_equalization.py
# Version:1.0
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
#图像灰度均衡化
img = cv2.imread("thumbsup.jpg", 1)
cv2.imshow("src", img)
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst_rgb", result)
cv2.imwrite("gray.png", result)
cv2.waitKey(0)


#图像均衡化处理后灰度图
img = cv2.imread("gray.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([gray],[0],None,[256],[0,256])
plt.figure()#新建一个图像
plt.title("Grayscale Histogram")
plt.xlabel("Bins")#X轴标签
plt.ylabel("# of Pixels")#Y轴标签
plt.plot(hist)
plt.xlim([0,256])#设置x坐标轴范围
plt.savefig("table.png")
plt.show()