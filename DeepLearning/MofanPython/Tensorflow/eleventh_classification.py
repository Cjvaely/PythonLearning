#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/6 09:52
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : eleventh_classification.py
# @Software: PyCharm
"""
分类问题
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data,MNIST a hand_write database
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def add_layer(inputs, in_size, out_size, activation_function=None,):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs


# 计算准确度
def compute_accuracy(v_xs, v_ys):
    # 全局变量
    global prediction
    # 生成预测值，生成一个1行10列的数组，每个位置对应数据的概率，概率越大说明是该数可能性大
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    # tf.argmax就是返回最大的那个数值所在的下标，如果是矩阵，那么返回的向量对应矩阵最大值元素
    # 预测值与实际值判断是否相等
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    # tf.cast 转换数据格式 将bool类型转换成float32的0 1数
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])    # 28x28个数据点
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
prediction = add_layer(xs, 784, 10,  activation_function=tf.nn.softmax)

# the error between prediction and real data
# cross_entropy与softmax搭配组成分类算法
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
# 梯度下降优化器，学习效率0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
# important step
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    # 取出一部分数据，100个100个学习，省时间效果还不错
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        # 打印准确度
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))
