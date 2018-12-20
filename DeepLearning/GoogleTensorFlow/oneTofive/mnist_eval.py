#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/24 22:01
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : mnist_eval.py
# @Software: PyCharm
# 这是mnist数据集的测试程序
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载mnist_inference.py和mnist中定义的常量和函数
from GoogleTensorFlow.oneTofive import mnist_inference, train_mnist

# 每10秒加载一次最新模型，并在测试数据上测试正确率
eval_interval_secs = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        # 定义输出的格式
        x = tf.placeholder(
            tf.float32, [None, mnist_inference.input_node],
            name='x-input'
        )
        y_ = tf.placeholder(
            tf.float32, [None, mnist_inference.input_node],
            name='y-input'
        )
        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }
        # 直接用封装的函数计算前向传播结果
        # 此处为测试程序，不关注正则化损失
        y = mnist_inference.inference(x, None)
        # 计算结果正确率。使用tf.argmax(y, 1)得到输入样例预测的类别
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名加载模型
        variable_averages = tf.train.ExponentialMovingAverage(
            train_mnist.moving_average_decay
        )
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # 每隔eval_interval_secs秒计算一次正确率
        while True:
            with tf.Session() as sess:
                # tf.train.get_checkpoint_state通过checkpoint文件找到目录中最新模型文件名
                ckpt = tf.train.get_checkpoint_state(
                    train_mnist.model_save_path
                )
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名得到模型保存的迭代轮数
                    global_step = ckpt.model_checkpoint_pat.split(
                        '/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy,
                                              feed_dict=validate_feed)
                    print('After %d training step(s), validation accuracy = %g' % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found!')
                    return


def main(argv=None):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    evaluate(mnist)
    return


if __name__ == '__main__':
    tf.app.run()