#!/usr/bin/env python
# -*- coding:utf-8 -*-

# !/usr/bin/env python
# encoding=utf-8
# -------------------------------------------------------------------------------
# Name:        rgbYcbcr
# Author:      xiezhanghua (xiezhanghua111@j163.com)
# Created:     2017/8/13下午9:23
# -------------------------------------------------------------------------------

import cv2
import numpy as np
from matplotlib import pyplot as plt

################################################################################

print 'Load Image'

imgFile = '/Users/xiezhanghua/PycharmProjects/learntf/vip/img/2.jpg'

# load an original image
img = cv2.imread(imgFile)
################################################################################

print 'YCbCr-RGB Skin Model'

rows, cols, channels = img.shape
################################################################################

# convert color space from rgb to ycbcr
imgYcc = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

# convert color space from bgr to rgb
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# prepare an empty image space
imgSkin = np.zeros(img.shape, np.uint8)
# copy original image
imgSkin = img.copy()
################################################################################

for r in range(rows):
    for c in range(cols):

        # non-skin area if skin equals 0, skin area otherwise
        skin = 0
        ########################################################################

        # get values from rgb color space
        R = img.item(r, c, 0)
        G = img.item(r, c, 1)
        B = img.item(r, c, 2)

        # get values from ycbcr color space
        Y = imgYcc.item(r, c, 0)
        Cr = imgYcc.item(r, c, 1)
        Cb = imgYcc.item(r, c, 2)
        ########################################################################

        # skin color detection

        if R > G and R > B:
            if (G >= B and 5 * R - 12 * G + 7 * B >= 0) or (G < B and 5 * R + 7 * G - 12 * B >= 0):
                if Cr > 135 and Cr < 180 and Cb > 85 and Cb < 135 and Y > 80:
                    skin = 1
                    # print 'Skin detected!'

        if 0 == skin:
            imgSkin.itemset((r, c, 0), 0)
            imgSkin.itemset((r, c, 1), 0)
            imgSkin.itemset((r, c, 2), 0)

# display original image and skin image
plt.subplot(1, 2, 1), plt.imshow(img), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(imgSkin), plt.title('Transformed YCbCr Skin Image'), plt.xticks([]), plt.yticks([])
plt.show()
################################################################################

print 'Goodbye!'