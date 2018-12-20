#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/14 11:04
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : CNN_cifar10.py
# @Software: PyCharm

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from TensorFlowPractice import cifar10
from TensorFlowPractice import cifar10_input
import tensorflow as tf
import numpy as np
import time

# 训练轮数
max_steps = 4000
batch_size = 128
# cifar10数据的缓存地址
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'


# 这里加了一个L2正则化处理，w1用于控制L2 loss的大小
def variable_with_weight_loss(shape, stddev, w1):
    # 定义初始weight，使用截断的正态分布初始化权重
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        # tf.nn.L2_loss计算weight的L2 loss
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        # 把weight loss加入到名为losses的collection中
        tf.add_to_collection('losses', weight_loss)
    return var


# 初次下载cifar10数据
# cifar10.maybe_download_and_extract()

# 获取训练样本
images_train, labels_train = cifar10_input.distorted_inputs(
    data_dir=data_dir, batch_size=batch_size
)
# 生成测试数据
images_test, labels_test = cifar10_input.inputs(eval_data=True,
                                         data_dir=data_dir,
                                         batch_size=batch_size
)
# 创建输入数据，特征数据和label
# 图片尺寸24x24 颜色为3
images_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

# 创建第一个卷积层
# 卷积核大小为5x5 3个颜色通道 64个卷积核
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2,
                                    w1=0.0)
# 使用conv2d对输入数据images_holder进行卷积，步长都是1
kernel1 = tf.nn.conv2d(images_holder, weight1, [1, 1, 1, 1], padding='SAME')
# 该卷继承的偏置为0
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
# ReLU激活函数进行非线性处理
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
# 最大池化尺寸3x3 步长3x3，尺寸和步长不一致，增加出库的数据丰富性
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding='SAME')
# LRN层处理，对于rel比较有用。但是不适合sigmoid
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# 创建第二个卷积层
# 卷积核输入通道是64
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2,
                                    w1=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
# 偏置为0.1
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
# relu激活函数处理
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
# 与第一层不同，先进行lrn处理
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
# 后进行最大池化
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding='SAME')

# 两个卷积层之后使用一个全连接层
# 第一层
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
# 全连接层的第一个隐藏层节点为384
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)
# 第二层
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
# 全连接层的第二个隐藏层节点为384/2
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
# 激活函数处理
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)
# 最后一层
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, w1=0.0)
# 最后一层节点数为10
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)


# 计算CNN的loss，使用的是交叉熵
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example'
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                        name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


# 导入logits和label_placeholder
loss = loss(logits, label_holder)
# 优化器
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# 启动线程队列加速
tf.train.start_queue_runners()

for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value = sess.run([train_op, loss],
                             feed_dict={images_holder: image_batch, label_holder: label_batch})
    duration = time.time() - start_time
    if step % 100 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)

        format_str = ('step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

num_examples = 10000
import math
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={images_holder: image_batch,
                                                 label_holder: label_batch})
    true_count += np.sum(predictions)
    step += 1

precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)