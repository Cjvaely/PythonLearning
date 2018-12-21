#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-21 11:12:19
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : http://www.cnblogs.com/cjvae/
# @Version : $Id$

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from simplepreprocessor import SimplePreprocessor
from simpledatasetloader import SimpleDatasetLoader
from imutils import paths
import argparse

if __name__ == '__main__':
	# 构造参数解析并解析参数
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", required=True,
		help= "path to input dataset")

	ap.add_argument("-k", "--neighbors", type=int, default=1,
		help="# of nearest neighbors for classification")

	ap.add_argument("-j", "--jobs", type=int, default=-1,
		help="# of jobs for k-NN distance (-1 uses all available cores)")
	args = vars(ap.parse_args())

	# 抓住我们将要描述的图像列表
	print("[INFO] loading images...")
	imagePaths = list(paths.list_images(args["dataset"]))

	# 初始化图像预处理器，从磁盘加载数据集，并重塑数据矩阵
	sp = SimplePreprocessor(32, 32)
	sdl = SimpleDatasetLoader(preprocessors=[sp])
	(data, labels) = sdl.load(imagePaths, verbose=100)
	data = data.reshape((data.shape[0], 3072))

	# 显示有关图内存消耗的一些信息
	print("[INFO] features matrix: {:.1f}MB".format(
		data.nbytes / (1024 * 1000.0)))

	# 将标签编码为整数
	le = LabelEncoder()
	labels = le.fit_transform(labels)

	# 使用75％的数据进行训练并将剩余的25％用于测试，将数据划分为训练和测试分组
	(trainX, testX, trainY, testY) = train_test_split(data, labels,
		test_size=0.25, random_state=42)
	 
	# 在原始像素强度上训练和评估k-NN分类器
	print("[INFO] evaluating k-NN classifier...")
	
	model = KNeighborsClassifier(n_neighbors=3)
	model.fit(trainX, trainY)
	print(classification_report(testY, model.predict(testX),
		target_names=le.classes_))
