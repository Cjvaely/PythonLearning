#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-21 02:16:19
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : http://www.cnblogs.com/cjvae/
# @Version : $Id$

import numpy as np
import cv2
import os

class SimpleDatasetLoader:
	def __init__(self, preprocessors=None):
		# 存储图像预处理器
		self.preprocessors = preprocessors

		# 如果预处理器为None，则将它们初始化为空列表
		if self.preprocessors is None:
			self.preprocessors = []

	def load(self, imagePaths, verbose=-1):
		# 初始化特征和标签列表
		data = []
		labels = []

		# 循环输入图像
		for (i, imagePath) in enumerate(imagePaths):
			# 加载图像并提取类标签，假设我们的路径具有以下格式：
			# /path/to/dataset/{class}/{image}.jpg
			image = cv2.imread(imagePath)
			label = imagePath.split(os.path.sep)[-2]
			
			# 检查我们的预处理器是否不是None
			if self.preprocessors is not None:
				# 循环预处理器并将每个应用于图像
				for p in self.preprocessors:
					image = p.preprocess(image)
					
			# 通过更新数据列表后跟标签，将处理后的图像视为“特征向量”
			data.append(image)
			labels.append(label)

			# 显示每个'verbose'图像的更新
			if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
				print("[INFO] processed {}/{}".format(
					i + 1, len(imagePaths)))

		# 返回数据和标签的元组
		return (np.array(data), np.array(labels))
		