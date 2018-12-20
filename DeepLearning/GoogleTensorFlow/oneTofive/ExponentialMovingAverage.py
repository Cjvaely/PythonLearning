#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/20 16:25
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : ExponentialMovingAverage.py
# @Software: PyCharm
# 实战tensorflow 4.4.3 滑动平均模型示例
import tensorflow as tf

# 定义一个变量用于计算滑动平均，这个变量初始值为0
v1 = tf.Variable(0, dtype=tf.float32)
# 这里step变量模拟神经网络中迭代轮数，动态控制衰减率
step = tf.Variable(0, trainable=False)

# 定义滑动平均类（class）。初始化时给定衰减率为0.99，控制衰减率变量step
ema = tf.train.ExponentialMovingAverage(0.99, step)
# 定义一个更新变量滑动平均的操作，每次执行操作，列表变量会更新
maintain_averages_op = ema.apply([v1])
with tf.Session() as sess:
    # 初始化变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 通过ema.average(v1)获滑动平均之后变量的值。初始化后，变量v1的值和v1的滑动平均都是0
    print('The first:Value and ExponentialMovingAverage_Value:')
    print(sess.run([v1, ema.average(v1)]))
    # 更新变量v1到5
    sess.run(tf.assign(v1, 5))
    # 更新v1的滑动平均值，衰减率为min{0.99, (1+step)/(10+step)=0.1}=0.1
    # 所以v1的滑动平均被更新到0.1x0 + 0.9 x 5 = 4.5
    sess.run(maintain_averages_op)
    print('The second:Value and ExponentialMovingAverage_Value:')
    print(sess.run ([v1, ema.average (v1)]))

    # 更新step的值为10000
    sess.run(tf.assign(step, 10000))
    # 更新v1的值为10
    sess.run(tf.assign(v1, 10))
    # 更新v1的滑动平均值，衰减率为min{0.99, (1+step)/(10+step)=0.999}=0.99
    # 所以v1的滑动平均被更新到0.99x0 + 0.01 x 10 = 4.555
    sess.run(maintain_averages_op)
    print('The third:Value and ExponentialMovingAverage_Value:')
    print(sess.run ([v1, ema.average (v1)]))

    # 再次更新滑动平均值，得到0.99 x 4.555 + 0.01 x 10 = 4.60945
    sess.run(maintain_averages_op)
    print('The fourth:Value and ExponentialMovingAverage_Value:')
    print(sess.run ([v1, ema.average (v1)]))