#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/26 18:39
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : Cnn_mnist_inference.py
# @Software: PyCharm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# 定义相关参数
input_node = 784
output_node = 10

image_size = 28
num_channels = 1
num_labels = 10

# 第一层卷积层的尺寸和深度
connv1_size = 5
conv1_deep = 32
# 全连接层的节点个数
fc_size = 512


# test
# 定义卷积神经网络的前向传播过程，添加参数train用于区分训练过程和测试过程
# 将用到dropout方法，可以进一步提升模型可靠性并防止过拟合
def inference(input_tensor, train, regularizer):
    # 声明第一层卷积神经网络的变量并完成前向传播过程
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = get_weight_variable(
            [input_node, layer1_node], regularizer
        )
        biases = tf.get_variable(
            'biases', [layer1_node],
            initializer=tf.constant_initializer(0.0)
        )
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    # 类似地声明第二层神经网络变量并完成前向传播过程
    with tf.variable_scope('layer2'):
        weights = get_weight1_variable(
            [layer1_node, output_node], regularizer
        )
        biases = tf.get_variable(
            'biases', [output_node],
            initializer=tf.constant_initializer(0.0)
        )
        layer2 = tf.matmul(layer1, weights) + biases

    # 返回最后的前向传播结果
    return layer2


