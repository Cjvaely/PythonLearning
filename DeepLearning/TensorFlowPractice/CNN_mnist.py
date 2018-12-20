#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/13 09:33
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : CNN_mnist.py
# @Software: PyCharm
# 实现简单的卷积神经网络
import os
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 句柄
sess = tf.InteractiveSession()


def weight_variable(shape):
    # 截断正态分布
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 2维卷积函数,x是输入，W是卷积参数[5, 5, 1, 32] 5x5 灰度单色1 32是卷积核量
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 2x2最大池化:将2x2像素块降为1x1。strides设置为横竖2步长
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


# x是特征
x = tf.placeholder(tf.float32, [None, 784])
# y_是真实数据
y_ = tf.placeholder(tf.float32, [None, 10])
# 将1x784的1D输入向量转化为2D图片，即把1x784转化成28x28，颜色通道为1
# 前面-1表示样本数量不固定
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 定义第一个卷积层
# 卷积核尺寸5x5 一个颜色 32个不同卷积核
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# conv2d进行卷积操作
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 对卷积的结果进行池化操作
h_pool1 = max_pool_2x2(h_conv1)

# 定义第二个卷积层。和第一个不同的是，卷积核的数量变成了64，会提取64种特征
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 两次步长为2x2的最大池化 边长变为——>1/4 7x7，但是此时卷积核数量为64
# 所以输出的tensor尺寸是7x7x64

# 第二层卷积层之后，连接一个全连接网络
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
# 将第二个卷积层的输出变成1D向量
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# 全连接层的隐藏层
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 防止过拟合，使用dropout
keep_drop = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_drop)

# 最后，将Dropout层的输出层与softmax连接，得到概率输出
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),
                                              reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 返回每行最大的索引值
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 初始化参数
tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1],
                                                  keep_drop: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_drop: 0.5})
print("test accuracy %g" % accuracy.eval(feed_dict={
    x: mnist.test.images, y_:mnist.test.labels, keep_drop: 1.0 }))
