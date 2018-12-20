#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-20 21:32:15
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : http://www.cnblogs.com/cjvae/
# @Version : $Id$

# Python和OpenCV中的基本图像处理：调整大小（缩放），旋转和裁剪

import cv2

# 加载图像并显示
# imread函数返回一个NumPy数组，表示图像本身
image = cv2.imread("jurassic-park-tour-jeep.jpg")
# 在屏幕上显示图像，参数是窗口的“名称”，从磁盘加载的图像
cv2.imshow("original", image)
# 调用waitKey暂停程序，直到我们按下键盘上的一个键
# 使用参数“0”表示任何按键将取消暂停执行
cv2.waitKey(0)

# 使用.shape查看图像尺寸 (388,647,3) 388行 647列 3通道
# 647像素宽 388像素高
print(image.shape)

# 将像素宽度设为100，需要记住纵横比，防止图像看起来歪曲
# 因此计算新图像与旧图像的比例
r = 100.0 / image.shape[1]
# 得到新的宽 高像素值
dim = (100, int(image.shape[0] * r))

#####################################################################
# 图像缩放函数														#
# cv2.resize(src,dsize,dst=None,fx=None,fy=None,interpolation=None) #
# 原图 输出图像尺寸 沿水平轴的比例因子 沿垂直轴的比例因子 插值方法			#
# INTER_AREA：使用像素区域关系进行重采样。								#
# 它可能是图像抽取的首选方法，因为它会产生无云纹理的结果。					#
# ###################################################################
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("resized", resized)
cv2.waitKey(0)

# 存储缩放图像
cv2.imwrite("resizeing.png", resized)

# 抓住图像的尺寸并计算图像的中心
(h, w) = image.shape[:2]
center = (w / 2, h / 2)

# #######################################################################
# 旋转图片180度															#
# cv.GetRotationMatrix2D(center, angle, scale, mapMatrix)				#
# 图像中心 旋转角度(正值表示逆时针旋转) 缩放因子(想将图像的大小减半，我们将使用0.5)	#
# #######################################################################
M = cv2.getRotationMatrix2D(center, 180, 1.0)
# 执行旋转
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("rotated", rotated)
cv2.waitKey(0)

# 存储旋转图像
cv2.imwrite("rotating.png", rotated)

# 使用数组切片裁剪图像 因为它是NumPy数组
cropped = image[70:170, 440:540]
cv2.imshow("cropped", cropped)
cv2.waitKey(0)

# 将裁剪后的图像保存到磁盘
cv2.imwrite("thumbnail.png", cropped)