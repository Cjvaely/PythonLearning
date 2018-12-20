#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/24 20:05
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : train_mnist.py
# @Software: PyCharm
# 这个程序是神经网络的训练程序，不再将训练代码和测试代码跑在一起
# 每1000轮输出一次batch上损失函数大小来估计训练效果
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载mnist_inference.py定义的常量和前向传播函数
from GoogleTensorFlow import mnist_inference

# 配置神经网络的参数
batch_size = 100
learning_rate_base = 0.8
learning_rate_decay = 0.99
regularaztion_rate = 0.0001
training_steps = 30000
moving_average_decay = 0.99

# 模型保存的路径和文件名
model_save_path = "/Users/cjv/Documents/deeplearningCode/GoogleTensorFlow/Models2/"
model_name = 'model1.ckpt'


def train(mnist):
    # 定义输出的placeholder
    x = tf.placeholder(
        tf.float32, [None, mnist_inference.input_node], name='x-input'
    )
    y_ = tf.placeholder(
        tf.float32, [None, mnist_inference.output_node], name='y-input'
    )

    regularizer = tf.contrib.layers.l2_regularizer(regularaztion_rate)
    # 直接使用inference中定义的前向传播过程
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均率、训练过程
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step
    )
    # apply()用于生成影子变量
    variable_averages_op = variable_averages.apply(
        tf.trainable_variables()
    )
    # 交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1)
    )
    # 平均交叉熵
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 损失函数
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        learning_rate_base,
        global_step,
        mnist.train.num_examples / batch_size,
        learning_rate_decay
    )
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化tensorflow持久化类
    saver =  tf.train.Saver()
    with tf.Session() as sess:
        init_op =  tf.global_variables_initializer()
        sess.run(init_op)
        # 在训练过程中年不再测试模型在验证数据上的表现。验证和测试会有一个独立程序
        for i in range(training_steps):
            xs, ys = mnist.train.next_batch(batch_size)
            _, loss_value, step = sess.run(
                [train_op, loss, global_step], feed_dict={x: xs, y_:ys})
            # 每1000轮保存一次模型
            if i % 1000 == 0:
                # 输出当前batch上损失函数的大小
                print('After %d training step(s),loss on training'
                      'batch is %g.' % (step, loss_value))
                # 保存当前模型，文件末尾加训练轮数
                saver.save(
                    sess, os.path.join(model_save_path, model_name),
                    global_step=global_step
                )


def main(argv=None):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    train(mnist)
    return 0


if __name__ == '__main__':
    tf.app.run()




