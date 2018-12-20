#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-20 22:20:30
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : http://www.cnblogs.com/cjvae/
# @Version : $Id$
# USAGE
# python histograms.py --image grant.jpg

from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2

# 解析参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
# 参数转化成字典形式存储
args = vars(ap.parse_args())

# 加载图像并显示
image = cv2.imread(args["image"])
cv2.imshow("image", image)

# 将图像转换为灰度并创建直方图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)

#####################################################################
# 构建图像直方图														#
# cv2.calcHist(images, channels, mask, histSize, ranges) 			#
# images：这是我们想要计算直方图的图像。把它包装成一个清单：[myImage]		#
# channels：索引列表，我们在其中指定要为其计算直方图的通道的索引。			#
# 要计算灰度图像的直方图，列表将是[0]。要计算所有三色，将是[0, 1, 2]			#
# mask：与原始图像形状相同的uint8图像									#
# histSize：表示这个直方图分成多少份										#
# ranges:直方图中各个像素的值,[0, 256],能表示像素值从0到256的像素			#
# ###################################################################
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])

# 抓取图像通道，初始化颜色元组，图形和平面特征向量
chans = cv2.split(image)
# OpenCV以相反的顺序将图像存储为NumPy数组：BGR
colors = ("b", "g", "r")

plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
# 初始化直方图列表
features = []

# 循环图像通道
for (chan, color) in zip(chans, colors):
	# 计算每个通道的直方图，将颜色直方图连接到我们的features列表
	hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
	features.extend(hist)

	# 画出直方图
	plt.plot(hist, color = color)
	plt.xlim([0, 256])
# ###################################################################
# 这里我们简单地展示了每个通道的扁平颜色直方图：							#
# 256个分区的维数 * 3个通道=768个总值									#
# 实际上，我们通常不会为每个通道使用256个分档，通常使用32-96个分箱之间的选择	#
# ###################################################################
print("扁平特征向量大小: %d" % (np.array(features).flatten().shape))

# 转到二维直方图 将直方图中的二进制数从256减少到32
# 可以更好地可视化结果
fig = plt.figure()

# 绘制绿色和蓝色的2D颜色直方图
ax = fig.add_subplot(131)

# 传递了两个频道的列表：绿色通道chans[1]和蓝色通道chans[0]
# 使用32个二进制位，而不是256个
hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None,
	[32, 32], [0, 256, 0, 256])

p = ax.imshow(hist, interpolation = "nearest")
ax.set_title("2D Color Histogram for Green and Blue")
plt.colorbar(p)

# 绘制绿色和红色的2D颜色直方图
ax = fig.add_subplot(132)

# 传递了两个频道的列表：绿色通道chans[1]和红色通道chans[2]
hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None,
	[32, 32], [0, 256, 0, 256])

p = ax.imshow(hist, interpolation = "nearest")
ax.set_title("2D Color Histogram for Green and Red")
plt.colorbar(p)

# 绘制蓝色和红色的2D颜色直方图
ax = fig.add_subplot(133)

# 传递了两个频道的列表：蓝色通道chans[0]和红色通道chans[2]
hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None,
	[32, 32], [0, 256, 0, 256])

p = ax.imshow(hist, interpolation = "nearest")
ax.set_title("2D Color Histogram for Blue and Red")
plt.colorbar(p)

# 检查一个2D直方图的维度
print("2D histogram shape: %s, with %d values" % (
	hist.shape, hist.flatten().shape[0]))

#####################################################################
# 我们的2D直方图只能考虑图像中3个通道中的2个,所以现在让我们构建一个3D颜色直方图 #
# 每个方向有8个区间,无法绘制3D直方图，理论与2D直方图完全相同					#
#####################################################################
hist = cv2.calcHist([image], [0, 1, 2],
	None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
print("3D histogram shape: %s, with %d values" % (
	hist.shape, hist.flatten().shape[0]))

# 显示,并等待
plt.show()
cv2.waitKey(0)