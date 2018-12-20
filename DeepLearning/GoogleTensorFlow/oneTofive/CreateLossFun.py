#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/19 21:22
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : CreateLossFun.py
# @Software: PyCharm
# 自定义损失函数
import tensorflow as tf
from numpy.random import RandomState
"""
这种情况出现在一个实际问题中：比如预测商家某个产品的销售量。
这个商品的成本是1元，但是利润是10元。
那么，少预测一个就少挣10元，多预测一个就少挣1元。
如果按照既定的最小化均方差，无法利润最大化。需要自己定义损失函数
"""
# 可以这样定义： loss = tf.reduce_sum(tf.where(tf.greater(v1, v2),
#     (v1-v2) * a, (v2 - v1) * b ))
# 即：如果v1大于v2，那么就得(v1-v2) * a的总和；否则，得(v2 - v1) * b 的总和

batch_size = 8

# 两个输入节点
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
# 回归问题一般只有一个输出节点(真实值)
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义了一个单层神经网络前向传播的过程，简单加权和
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y  = tf.matmul(x, w1)

# 定义预测销售量多了还是少了的成本
loss_less = 10
loss_more = 1
loss = tf.reduce_sum(tf.where(
    tf.greater(y, y_),
    (y  - y_) * loss_more,
    (y_ - y) * loss_less
))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# 设置回归的正确值为两个输入的和加上一个随机变量
# 加随机变量就是加入不可预测的噪音，一般噪音是均值为0的小量
# 这里设置为-0.05-0.05的随机数
Y = [[x1 + x2 + rdm.rand() / 10.0 - 0.05] for (x1, x2) in X]

# 训练神经网络
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)
        sess.run(train_step,
                 feed_dict={x: X[start:end], y_: Y[start:end]})
        print(sess.run(w1))
"""
当loss_less = 1  loss_more = 10时，w1最终结果是：
[[ 1.01934695]
 [ 1.04280889]]
 此时，倾向于预测多一点 
 
当loss_less = 10  loss_more = 1时，w1最终结果是：
[[ 0.95525807]
 [ 0.9813394 ]]
 此时，倾向于预测少一点 
"""