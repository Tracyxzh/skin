#!/usr/bin/env python
# -*- coding:utf-8 -*-

# !/usr/bin/env python
# encoding=utf-8
# -------------------------------------------------------------------------------
# Name:        Ycbcr
# Author:      xiezhanghua (xiezhanghua111@j163.com)
# Created:     2017/8/13下午8:52
# -------------------------------------------------------------------------------


import numpy as np
import cv2
from matplotlib import pyplot as plt

test_image = cv2.imread('/Users/xiezhanghua/PycharmProjects/learntf/vip/img/2.jpg')

# # RGB到YCbCr色彩空间
# image_YCbCr = cv2.cvtColor(test_image, cv2.COLOR_RGB2YCrCb)
#
# y, cb, cr = cv2.split(image_YCbCr)
#
# plt.subplot(131), plt.imshow(y)
# plt.subplot(132), plt.imshow(cb)
# plt.subplot(133), plt.imshow(cr)
img = test_image
# 分通道计算每个通道的直方图
hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])


# 定义Gamma矫正的函数
def gamma_trans(img, gamma):
    # 具体做法是先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)

    # 实现这个映射用的是OpenCV的查表函数
    return cv2.LUT(img, gamma_table)


# 执行Gamma矫正，小于1的值让暗部细节大量提升，同时亮部细节少量提升
img_corrected = gamma_trans(img, 0.5)
# cv2.imwrite('gamma_corrected.jpg', img_corrected)
plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(img_corrected)
# # 分通道计算Gamma矫正后的直方图
# hist_b_corrected = cv2.calcHist([img_corrected], [0], None, [256], [0, 256])
# hist_g_corrected = cv2.calcHist([img_corrected], [1], None, [256], [0, 256])
# hist_r_corrected = cv2.calcHist([img_corrected], [2], None, [256], [0, 256])
#
# fig = plt.figure()
#
# pix_hists = [
#     [hist_b, hist_g, hist_r],
#     [hist_b_corrected, hist_g_corrected, hist_r_corrected]
# ]
#
# pix_vals = range(256)
# for sub_plt, pix_hist in zip([121, 122], pix_hists):
#     ax = fig.add_subplot(sub_plt, projection='3d')
#     for c, z, channel_hist in zip(['b', 'g', 'r'], [20, 10, 0], pix_hist):
#         cs = [c] * 256
#         ax.bar(pix_vals, channel_hist, zs=z, zdir='y', color=cs, alpha=0.618, edgecolor='none', lw=0)
#
#     ax.set_xlabel('Pixel Values')
#     ax.set_xlim([0, 256])
#     ax.set_ylabel('Channels')
#     ax.set_zlabel('Counts')

plt.show()