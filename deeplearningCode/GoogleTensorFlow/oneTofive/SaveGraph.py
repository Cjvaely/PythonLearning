#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/23 21:48
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : SaveGraph.py
# @Software: PyCharm
# 一个小例子 保存tensorflow的计算图
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# 声明两个变量计算和
v1 = tf.Variable(tf.constant(1., shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(2., shape=[1]), name='v2')
result = v1 + v2

# 变量初始化
init_op = tf.global_variables_initializer()
# 声明tf.train.Saver类用于保存模型
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    # 将模型保存到./Models内
    saver.save(sess, '/Users/cjv/Documents/deeplearningCode/GoogleTensorFlow/Models/model1.ckpt')
# 生成的第一个文件是：model1.ckpt.meta    记录计算图结构
# 生成的第二个文件是：model1.ckpt 记录每一个变量的取值
# checkpoint文件保存一个目录下的所有模型文件列表