#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/22 11:22
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : MNIST_datatype.py
# @Software: PyCharm
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集，指定数据集地址
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 打印Training data size:55000
print("Training data size: ", mnist.train.num_examples)

# 打印Validating data size:5000
print("Validating data size: ", mnist.validation.num_examples)

# 打印Testing data size: 10000
print("Testing data size: ", mnist.test.num_examples)

# 打印Example training data:
print("Example training data: ", mnist.train.images[0])

# 打印Example training data label:
print("Example training data label: ", mnist.train.labels[0])

# --------------------------------------------------------------
# 从上可知 通过函数 input_date.read_data_sets 生成的类会自动地将MNIST数据
# 分成train（55000） validation（5000）这两个训练数据集
# test（10000）这个测试数据集。
# 经过处理，每个图片成为长度为784（28x28）的一维数组。数组中的元素对应图片像素矩阵
# 每一个数字。这样就把一张二维图像的像素矩阵放到一个一维数组中。像素矩阵中元素取值为[0-1]
# 代表颜色深浅。0表示白，1表示黑
# input_date.read_data_sets函数还提供了一个mnist.train.next_batch函数，
# 从训练数据读取一小部分作为训练batch
batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)
# 从train集合中选取batch_size个训练数据
print('X Shape:', xs.shape)
# 输出X shape：（100， 784）
print('Y Shape:', ys.shape)
# 输出Y shape：(100, 10)
