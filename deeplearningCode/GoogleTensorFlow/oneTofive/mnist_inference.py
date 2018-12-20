#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/24 17:21
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : mnist_inference.py
# @Software: PyCharm
# 该程序定义了前向传播过程以及神经网络中的参数
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# 定义相关参数
input_node = 784
output_node = 10
layer1_node = 500

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
        'weights', shape,
        initializer=tf.truncated_normal_initializer(stddev=0.1)
    )
    if regularizer != None:
        # 该函数的作用是将一个张量加入一个集合
        tf.add_to_collection('losses', regularizer(weights))
        return weights


# 定义神经网络前向传播过程
def inference(input_tensor, regularizer):
    # 声明第一层神经网络的变量并完成前向传播过程
    with tf.variable_scope('layer1'):
        weights = get_weight_variable(
            [input_node, layer1_node], regularizer
        )
        biases = tf.get_variable(
            'biases', [layer1_node],
            initializer=tf.constant_initializer(0.0)
        )
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    # 类似地声明第二层神经网络变量并完成前向传播过程
    with tf.variable_scope('layer2'):
        weights = get_weight_variable(
            [layer1_node, output_node], regularizer
        )
        biases = tf.get_variable(
            'biases', [output_node],
            initializer=tf.constant_initializer(0.0)
        )
        layer2 = tf.matmul(layer1, weights) + biases

    # 返回最后的前向传播结果
    return layer2


