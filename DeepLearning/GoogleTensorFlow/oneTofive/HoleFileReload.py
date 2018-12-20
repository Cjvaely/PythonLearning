#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/24 15:29
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : HoleFileReload.py
# @Software: PyCharm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    model_filename = "/Users/cjv/Documents/deeplearningCode/GoogleTensorFlow/Models/combinded_model.pb"
    # 读取保存的模型文件，将文件解析成对应的GraphDef Protocol Buffer
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 将graph_def中保存的图加载到当前图，return_elements=['add:0']给出
    # 返回的张量名称。保存时是计算节点的名称，所以加载的张量名称是add:0
    result = tf.import_graph_def(graph_def, return_elements=['add:0'])
    print(sess.run(result))
    # 输出是：[array([ 3.], dtype=float32)]