#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/7 10:59
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : 13_cnn.py
# @Software: PyCharm
"""
卷积神经网络识别手写图片
参考莫烦python

流行的搭建结构（从下到上依次是）：
输入数据（一张图片）——>一层卷积层——>池化（Max Pooling）——>两层全连接神经层——>分类器

隐藏层包含卷积层和pooling层
卷积——>把图像长宽压缩，厚度增加

patch/kernel是图像中的一部分，有自己的长宽，会被抽离
参数stride 表示每次跨stride个像素点抽离分析

多个patch抽离后合并 组成一个比原来图像小但是厚度大的立方体
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
# one hot 一位有效编码
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


# 权重参数
def weight_variable(shape):
    # tf.truncated_normal从截断的正态分布中输出随机值，
    # 生成的值服从具有指定平均值和标准偏差的正态分布
    # 如果生成的值大于平均值2个标准偏差的值则丢弃重新选择
    # tf.truncated_normal (shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
    # stddev正态分布的标准差
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 偏置值
def bias_variable(shape):
    # 定义常量0.1
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 二维的CNN
def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1，第二个1表示x方向，第三个是y方向
    # SAME抽取的是和原图片规格一样的，而VALID比原图片略小
    # 参数x就是整张输入图片的信息
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# pooling 池化 使得显示更多图像信息
def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])        # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
# 改变xs的形状,图片颜色是黑白 所以为1 如果彩色，为3
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# print(x_image.shape)    # [n_samples]

# conv1 layer ##
W_conv1 = weight_variable([5, 5, 1, 32])    # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)    # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)                             # output size 14x14x32

# conv2 layer ##
W_conv2 = weight_variable([5, 5, 32, 64])    # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)    # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                             # output size 7x7x64

# func1 layer ##
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# func2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
# important step
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000]))
