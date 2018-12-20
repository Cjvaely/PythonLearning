#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/16 09:51
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : test.py
# @Software: PyCharm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

# a = tf.constant([1 , 2 , 3])
# b = tf.constant([4 , 5 , 6])
#
# c = tf.stack([a , b] , axis=0)
# d = tf.unstack(c , axis=0)
# e = tf.unstack(c , axis=1)
#
# x = tf.constant([[1, 2, 3], [4, 5, 6]])
# x1 = tf.transpose(x)
# x2 = tf.transpose(x, perm=[1, 0])
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(c))
#     print(sess.run(d))
#     print(sess.run(e))
#     print(sess.run(x1))
#     print (sess.run (x2))


# labels = [[0.2, 0.3, 0.5],
#           [0.1, 0.6, 0.3]]
# logits = [[4, 1, -2],
#           [0.1, 1, 3]]
#
# logits_scaled = tf.nn.softmax (logits)
# result = tf.nn.softmax_cross_entropy_with_logits (labels=labels, logits=logits)
#
# with tf.Session () as sess:
#     print (sess.run (logits_scaled))
#     print (sess.run (result))

# a = [[1, 2, 3],
#       [2, 3, 4],
#       [5, 4, 3],
#       [8, 7, 2]]
# a0 = tf.argmax(a, 0)
# a1 = tf.argmax(a, 1)
# with tf.Session() as sess:
#     print(sess.run(a0))
#     print(sess.run(a1))

# matrix1 = tf.constant([[3, 3]])
# matrix2 = tf.constant([[2],
#                        [2]])
# with tf.Session() as sess:
#     print(sess.run (matrix1))
#     print(sess.run (matrix2))

# tf.variable_scope函数是可以嵌套的，如下所示
# with tf.variable_scope('root'):
#     # 可以通过tf.variable_scope().reuse函数获取上下文管理器中reuse取值
#     print(tf.get_variable_scope().reuse)            # False
#
#     with tf.variable_scope('foo', reuse=True):
#         print(tf.get_variable_scope().reuse)        # True
#         with tf.variable_scope('bar'):
#             print(tf.get_variable_scope().reuse)    # True
#     print (tf.get_variable_scope ().reuse)          # False

# tf.variable_scope()管理变量名称
# v1 = tf.get_variable('v', [1])
# # 输出 v:0;'v'表示变量名称':0'表示这个变量时生成的第一个结果
# print(v1.name)
#
# with tf.variable_scope('foo'):
#     v2 = tf.get_variable('v', [1])
#     # 输出 foo/v:0;在tf.variable_scope中创建的变量，变量名称前带空间名称
#     # 以'/'作为分隔
#     print(v2.name)
#
# with tf.variable_scope('foo'):
#     with tf.variable_scope('bar'):
#         v3 = tf.get_variable('v', [1])
#         # 输出 foo/bar/v:0;命名空间可以嵌套
#         print (v3.name)
#     v4 = tf.get_variable('v1', [1])
#     # foo/v1:0;退出命名空间后就不会加前缀了
#     print(v4.name)
# with tf.variable_scope('', reuse=True):
#     v5 = tf.get_variable('foo/bar/v', [1])
#     # 输出为：True；可以通过带命名空间的变量名查找其他命名空间的变量
#     print(v5 == v3)
#     v6 = tf.get_variable('foo/v1', [1])
#     print(v6 == v4)

# 保存滑动平均模型
v = tf.Variable(0, dtype=tf.float32, name='v')
# 此时还没声明滑动平均模型，只有变量'v'，所以输出："v:0"
for variables in tf.global_variables():
    print(variables.name)
ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.global_variables())
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    sess.run(tf.assign(v, 10))
    sess.run(maintain_averages_op)
    # 保存时，tensorflow会将v:0和v/ExponentialMovingAverage:0都保存
    saver.save(sess, '/Users/cjv/Documents/deeplearningCode/GoogleTensorFlow/Models/model2.ckpt')
    # 输出的[10.0, 0.099999905],后者就是v的滑动平均值
    print(sess.run([v, ema.average(v)]))
