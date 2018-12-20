#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/2 15:42
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : fourth_placeholder.py
# @Software: PyCharm

"""
my fourth deeplearning code

placeholder 是 Tensorflow 中的占位符，暂时储存变量
Tensorflow 如果想要从外部传入data, 那就需要用到 tf.placeholder()

总结：tf.placeholder()作用就是暂时存储变量，用于获取输入值
    再根据输入值进行某些处理，获得输出值
    其中需要用到tensorflow中的字典feed_dict
"""
from __future__ import print_function
import tensorflow as tf

# 在 Tensorflow 中需要定义 placeholder的type ，一般为 float32 形式
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# mul = multiply 是将input1和input2 做乘法运算，并输出为 output
output = tf.multiply(input1, input2)

# sess.run()用于传值，需要传入的值放在了feed_dict={}
# 一一对应每一个 input: placeholder
with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))


'''
tensorflow: 1.1.0
'''
x1 = tf.placeholder(dtype=tf.float32, shape=None)
y1 = tf.placeholder(dtype=tf.float32, shape=None)
z1 = x1 + y1

x2 = tf.placeholder(dtype=tf.float32, shape=[2, 1])
y2 = tf.placeholder(dtype=tf.float32, shape=[1, 2])
z2 = tf.matmul(x2, y2)

with tf.Session() as sess:
    # when only one operation to run
    z1_value = sess.run(z1, feed_dict={x1: 1, y1: 2})

    # when run multiple operations, run them together
    z1_value, z2_value = sess.run(
        [z1, z2], feed_dict={
            x1: 1, y1: 2,
            x2: [[2], [2]], y2: [[3, 3]]
        })
    print(z1_value)
    print(z2_value)
