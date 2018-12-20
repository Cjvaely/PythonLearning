#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/12 16:39
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : tf_MLP.py
# @Software: PyCharm
# 用tensorflow实现多层感知机（Multi—Layer Perceptron）

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 句柄
sess = tf.InteractiveSession()

# 输入节点数
in_units = 784
# 隐藏层的输出节点数（200-1000都可）
h1_units = 300
# 隐藏层权重,标准差为0.1正态分布值,形状为784行，300列
w1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
# 隐藏层的偏置
b1 = tf.Variable(tf.zeros([h1_units]))
w2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, in_units])
# dropout的比率（保留节点的概率定义）
keep_prob = tf.placeholder(tf.float32)

# 定义模型结构
# 激活函数为ReLU的隐藏层
hiddlen1 = tf.nn.relu(tf.matmul(x, w1) + b1)
# tf.nn.dropout：随机将一部分点置为0，keep_prob是不置为0的比例
# 这是用于制造随机性，防止过拟合的
hiddlen1_drop = tf.nn.dropout(hiddlen1, keep_prob)
# 输出层，选出最可能的类
y = tf.nn.softmax(tf.matmul(hiddlen1_drop, w2) + b2)
# 实际数据
y_ = tf.placeholder(tf.float32, [None, 10])
# 定义损失函数和优化器优化loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                              reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
# train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
# 开始训练
tf.global_variables_initializer().run()
# 选择3000个batch
for i in range(3000):
    # 每个batch为100条数据
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs, y_:batch_ys, keep_prob:0.75})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    if i % 100 == 0:
        print("Step %d," % i, accuracy.eval({x:mnist.test.images, y_:mnist.test.labels,
                         keep_prob:1.0}))
        # AdagradOptimizer:Step 2900, 0.9801
        # AdamOptimizer: Step 2900, 0.9754

