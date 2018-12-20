#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/7 16:56
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : 15_rnn_classification.py
# @Software: PyCharm
"""
使用循环神经网络，导入手写数字MNIST数据集来进行分类例子的训练
关于LSTM（长短期记忆）
三个控制器：输入控制 输出控制 忘记控制
参考：莫烦python
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# 导入MNIST数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyper parameters（超参数）：在学习过程开始之前设置其值的参数。
# 相比之下，其他参数的值是通过训练得出的。

# 学习速率
lr = 0.001
# 训练步数的上限
training_iters = 100000
batch_size = 128

n_inputs = 28   # MNIST数据集输入 (图片尺寸: 28*28)
n_steps = 28    # time steps
n_hidden_units = 128    # 隐藏层的神经元数量，自己设置
n_classes = 10          # MNIST 数据集的类别总量 (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# 对 weights biases 初始值的定义
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


# 主体结构 3个组成部分（input_layer，cell，output_layer）
def RNN(X, weights, biases):
    # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
    # X ==> (128 batches * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])   # 变成2维

    # into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)    换回3维
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    ##########################################

    # basic LSTM Cell.

    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    # lstm cell 可以被分为两种状态 (c_state, h_state)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    # You have 2 options for following step.
    # 1: tf.nn.rnn(cell, inputs);
    # 2: tf.nn.dynamic_rnn(cell, inputs).
    # If use option 1, you have to modified the shape of X_in, go and check out this:
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
    # In here, we go for option 2.
    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    # Make sure the time_major is changed accordingly.
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    # hidden layer for output as the final results
    #############################################
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # # or
    # unpack to list [(batch, outputs)..] * steps
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)

    return results


# 训练神经网络
pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

# 比较训练值（pred）与真实值(y)是否相等
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# tf.cast转换数据格式
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    # 初始化所有变量
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 一个图片只有一串数据，通过reshape可以把它变成28行28列数组
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
            }))
        step += 1




