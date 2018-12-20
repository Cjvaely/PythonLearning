#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/24 00:03
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : ReloadSaveGraph.py
# @Software: PyCharm
# 加载已保存的tensorflow模型
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# 使用和保存模型一样的方式声明变量
v1 = tf.Variable(tf.constant(1., shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(2., shape=[1]), name='v2')
result = v1 + v2

saver = tf.train.Saver()

with tf.Session() as sess:
    # 加载已经保存的模型，通过已保存的模型中变量值计算加法
    saver.restore(sess, '/Users/cjv/Documents/deeplearningCode/GoogleTensorFlow/Models/model1.ckpt')
    print(sess.run(result))
