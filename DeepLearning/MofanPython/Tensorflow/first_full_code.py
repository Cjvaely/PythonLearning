#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/2 10:47
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : first_full_code.py
# @Software: PyCharm
""""
my first deeplearning code
"""

from __future__ import print_function
import tensorflow as tf
import numpy as np

# 使用numpy创建数据
# 随机生成和的100个[0, 1)之间的浮点数数组 数据类型是float32，由于tensorflow中数据类型主要就是float32
x_data = np.random.rand(100).astype(np.float32)

# 需要通过学习得到的目标数Wight——>0.1 biases——>0.3
y_data = x_data*0.1 + 0.3

# 开始创建tensorflow结构

# Weight(该数字的目标就是0.1)
# tf.Variable参数变量:Weight的初始数据，是随机生成的一维数据，范围是-1.0~1.0
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# biases（该数的目标是0.3）
# 初始定义为0
biases = tf.Variable(tf.zeros([1]))

# 定义预测数据y
y = Weights*x_data + biases

# 预测的y值与目标值的差
loss = tf.reduce_mean(tf.square(y-y_data))

# optimizer优化器，减少误差,反向传递误差
# GradientDescentOptimizer 梯度下降优化器，0.5是学习效率，一般小于1
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 结束创建tensorflow结构

# 创建会话,用 Session 来执行 init 初始化步骤,用Session来 run 每一次 training 的数据. 逐步提升神经网络的预测准确性.
sess = tf.Session()

# 初始化所有之前定义的Variable，即初始化结构，使结构激活
init = tf.global_variables_initializer()
# 激活session，很重要
sess.run(init)

# 训练201次
for step in range(201):
    sess.run(train)     # 开始训练
    if step % 20 == 0:      # 每20次打印一次
        print(step, sess.run(Weights), sess.run(biases))

"""训练结果
Weight初始值0.19553389，逐渐趋近0.1，最后值是0.10000013
biases初始值为0.32357454，逐渐趋近0.3，最后值是0.29999995
0 [0.19553389] [0.32357454]
20 [0.11840957] [0.29123077]
40 [0.10493226] [0.29765058]
60 [0.10132144] [0.29937056]
80 [0.10035404] [0.29983136]
100 [0.10009485] [0.29995483]
120 [0.10002542] [0.2999879]
140 [0.10000681] [0.29999676]
160 [0.10000181] [0.29999915]
180 [0.10000048] [0.29999977]
200 [0.10000013] [0.29999995]
"""
