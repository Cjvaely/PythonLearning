#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/23 21:28
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : MNIST_Train2.py
# @Software: PyCharm
# 相比MNIST_Train.py 这个版本优化了inference函数
# 不需要态度变量传入，提高可读性

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST 数据集相关的常数
input_node = 784    # 输入层的节点数，对于MNIST就是图片像素数
output_node = 10    # 输出层的节点数，对于MNIST就是类别数目0-9

# 配置神经网络的参数
layer1_node = 500   # 隐藏层节点数500，这里使用只有一个的隐藏层的网络结构
batch_size = 100    # 一个训练batch中训练数据的个数
                    # 数字越小，训练过程越接近的随机梯度下降；越大接近梯度下降
learning_rate_base = 0.8    # 基础学习率
learning_rate_decay = 0.99  # 学习率的衰减率

regularization_rate = 0.0001    # 描述模型复杂度的正则化项在损失函数中的系数
training_steps = 200000          # 训练轮数
moving_average_decay = 0.99     # 滑动平均衰减率

def inference(input_tensor, reuse=False):
    # 定义第一层神经网络的变量和前向传播过程
    with tf.variable_scope ('layer1', reuse=reuse):
        # 根据reuse判断是新变量还是已创建的，第一次构造是新变量，后面是reuse=True
        weights = tf.get_variable('weights', [layer1_node, output_node],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [output_node],
                                 initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

        # 定义第二层神经网络和前向传播过程
    with tf.variable_scope ('layer2', reuse=reuse):
        weights = tf.get_variable ('weights', [layer1_node, output_node],
                                       initializer=tf.truncated_normal_initializer (stddev=0.1))
        biases = tf.get_variable ('biases', [output_node],
                                      initializer=tf.constant_initializer (0.0))
        layer2 = tf.matmul (layer1, weights) + biases
    return layer2

x = tf.placeholder(tf.float32, [None, input_node], name='x-input')
y = inference(x)

# 之后创建变量
new_x = ...
new_y = inference(new_x, True)


