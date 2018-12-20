#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/2 16:11
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : fifth_activation.py
# @Software: PyCharm
"""
激励函数：先激活一部分神经元


"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 制造数据，定义x数据范围（-5，5）个数200个
x = np.linspace(-5, 5, 200)     # x data, shape=(100, 1)
'''
X 数据大致如此
array([-5.        , -4.94974874, -4.89949749, -4.84924623, -4.79899497,
       -4.74874372, -4.69849246, -4.64824121, -4.59798995, -4.54773869,
       -4.49748744, -4.44723618, -4.39698492, -4.34673367, -4.29648241,
       -4.24623116, -4.1959799 , -4.14572864, -4.09547739, -4.04522613,
       -3.99497487, -3.94472362, -3.89447236, -3.84422111, -3.79396985,
       -3.74371859, -3.69346734, -3.64321608, -3.59296482, -3.54271357,
       -3.49246231, -3.44221106, -3.3919598 , -3.34170854, -3.29145729,
       -3.24120603, -3.19095477, -3.14070352, -3.09045226, -3.04020101,
'''

# 下面是一些常用的激励函数
y_relu = tf.nn.relu(x)
y_sigmoid = tf.nn.sigmoid(x)
y_tanh = tf.nn.tanh(x)
y_softplus = tf.nn.softplus(x)
# softmax is a special kind of activation function, it is about probability
# y_softmax = tf.nn.softmax(x)

sess = tf.Session()
y_relu, y_sigmoid, y_tanh, y_softplus = sess.run([y_relu, y_sigmoid, y_tanh, y_softplus])

# 激励函数可视化

# 生成一个窗口， 大小8x6
plt.figure(1, figsize=(8, 6))
# 组合许多的小图, 放在一张大图里面显示的，221表示窗口分为两行两列，当前位置为1
plt.subplot(221)
# 画图x和y的函数图像，线颜色为红色，标签relu
plt.plot(x, y_relu, c='red', label='relu')
# y轴范围（-1，5）
plt.ylim((-1, 5))
# 标签加在右上角
plt.legend(loc='best')

# 窗口两行两列 当前位置为2
plt.subplot(222)
plt.plot(x, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

# 窗口两行两列 当前位置为3
plt.subplot(223)
plt.plot(x, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

# 窗口两行两列 当前位置为4
plt.subplot(224)
plt.plot(x, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()
