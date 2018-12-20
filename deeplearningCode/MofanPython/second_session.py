#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/2 14:56
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : second_session.py
# @Software: PyCharm
"""
my second deeplearning code

Session是Tensorflow为了控制、输出文件的执行的语句
运行 session.run() 可以获得你要得知的运算结果或者是你所要运算的部分
"""
from __future__ import print_function
import tensorflow as tf

# 建立两个矩阵
matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2],
                       [2]])

# numpy中的矩阵乘法：np.dot(m1, m2)
# product 不是直接计算的步骤
# 这是tensorflow计算矩阵乘法的方法
product = tf.matmul(matrix1, matrix2)

# 打开session方法一
sess = tf.Session()
result1 = sess.run(product)
print("Method 1：", result1)
sess.close()

# 打开session方法2 with语句 自动关闭session
with tf.Session() as sess:
    result2 = sess.run(product)
    print("Method 2：", result2)


'''
下面是基于tensorflow: 1.1.0的版本
'''
# 建立矩阵
m1 = tf.constant([[2, 2]])
m2 = tf.constant([[3],
                  [3]])

dot_operation = tf.matmul(m1, m2)

# wrong! no result
print("dot_operation：", dot_operation)

# method1 use session
sess = tf.Session()
result_1 = sess.run(dot_operation)
print(result_1)
sess.close()

# method2 use session
with tf.Session() as sess:
    result_2 = sess.run(dot_operation)
    print(result_2)
