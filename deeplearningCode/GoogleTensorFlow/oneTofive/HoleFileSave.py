#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/24 14:34
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : HoleFileSave.py
# @Software: PyCharm
# 通过函数将计算图中的变量以及取值通过常量方式保存

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.python.framework import graph_util

v1 = tf.Variable(tf.constant(1., shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(2., shape=[1]), name='v2')
result = v1 + v2

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    # 导出计算图的GraphDef部分，只需这部分可以完成输入层到输出层计算
    graph_def = tf.get_default_graph().as_graph_def()

    # 将图中变量及其取值转化成常量，将图中不必要节点去掉
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph_def, ['add']
    )
    # 将导出的模型存入文件
    with tf.gfile.GFile("/Users/cjv/Documents/deeplearningCode/GoogleTensorFlow/Models/combinded_model.pb", 'wb')as f:
        f.write(output_graph_def.SerializeToString())