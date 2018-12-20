#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/19 15:35
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : FirstNN.py
# @Software: PyCharm
# 实战谷歌框架第三章  完整示例训练神经网络
import tensorflow as tf
from numpy.random import RandomState

# 定义训练的数据集大小
batch_size = 8
# 定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
# y_是正确的结果
y_=tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义神经网络的前向传播过程
a = tf.matmul(x, w1)
# 预测的结果
y = tf.matmul(a, w2)

# 定义损失函数:刻画预测值和真实值的差距
# 和反向传播算法
# 交叉熵的计算方法：H(p, q),其中的p是正确值，q是预测值。
# 例如正确值是（1，0，0）预测值是（0.8，0.1，0.1）
# 那么交叉熵是 -（1 * log0.8 + 0 * log0.1 + 0 * log0.1）
# 交叉熵越小，预测答案越准确
# tf.clip_by_value将张量中的数值限定在一定的范围内
cross_entropy = -tf.reduce_mean(
    y * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 通过随机数生成一个模拟数据
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# 在这里定义规则，选择标签。所有的x1+x2<1被认为是正样本，否则是负样本。
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

# 初始化变量
init_op = tf.global_variables_initializer()
# 创建一个会话运行程序
with tf.Session() as sess:
    # 激活变量
    sess.run(init_op)
    print('The value of Weight before training...')
    print(sess.run(w1))
    print(sess.run (w2))
    # """
    # 训练之前的神经网络参数
    # [[-0.81131822  1.48459876  0.06532937]
    #  [-2.44270396  0.0992484   0.59122431]]
    #
    # [[-0.81131822]
    #  [ 1.48459876]
    #  [ 0.06532937]]
    # """
    STEPS = 5000
    for i in range(STEPS):
        # 没次选取batch_size个样本训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        # 通过选取的样本训练神经网络并更新参数
        sess.run(train_step,
                 feed_dict={x:X[start:end], y_:Y[start:end]})
        if i % 1000 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(
                cross_entropy, feed_dict={x:X, y_:Y})
            print("After %d training step(s), cross entropy on all data is %g" %
                  (i, total_cross_entropy))
    print('The value of Weight After training...')
    print(sess.run (w1))
    print(sess.run (w2))
    # """
    # 训练之后的神经网络参数
    # [[-1.59736061  2.24021816  1.12198699]
    #  [-3.05371118  0.68570381  1.44098806]]
    # [[-1.4770813 ]
    #  [ 2.28741789]
    #  [ 0.90313989]]
    # """
# 可以看到，参数前后是不同的。这就是神经网络训练的结果，训练后更高地拟合数据