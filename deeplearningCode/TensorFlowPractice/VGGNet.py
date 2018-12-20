#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018-11-25 11:17
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : VGGNet.py
# @Software: PyCharm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from  datetime import datetime
import math
import  time
import  tensorflow as tf

batch_size = 32
num_batches = 100


# 创建卷积层并保存本层参数
# 参数依次是：输入tensor，这层名称，kernel height（卷积核高），kernel width卷积核宽
# 卷积核数量即通道数，dh是步长高，dw是步长宽，p是参数列表
def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    n_in = input_op.get_shape()[-1].value


    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w',
                                 shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        # 卷积处理 步长dh x dw
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1),
                            padding='SAME')
        # biases为0
        bisa_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        # 转换成训练参数
        biases = tf.Variable(bisa_init_val, trainable=True, name='b')
        # 卷积结果与biases相加
        z = tf.nn.bias_add(conv, biases)
        # 非线性处理
        activation = tf.nn.relu(z, name=scope)
        # 添加参数进参数列表
        p += [kernel, biases]
        return activation


# 全连接层创建函数fc_op
def fc_op(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value


    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w',
                                 shape=[n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[n_out],
                                         dtype=tf.float32, name='b'))
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        p += [kernel, biases]
        return activation


# 最大池化
def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(
        input_op,
        ksize=[1, kh, kw, 1],
        strides=[1, dh, dw, 1],
        padding='SAME',
        name=name
    )


# VGGNet16主要分为6部分：前5段是卷积网络 最后一段是全连接网络
def inference_op(input_op, keep_prob):
    p = []

    # 第一段由两个卷积一个池化组成
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64,
                      dh=1, dw=1, p=p)

    conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64,
                      dh=1, dw=1, p=p)
    pool1 = mpool_op(conv1_2, name="pool1", kh=2, kw=2, dh=2, dw=2)

    # 第二段
    conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3,
                      n_out=128, dh=1, dw=1, p=p)
    conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128,
                      dh=1, dw=1, p=p)
    pool2 = mpool_op(conv2_2, name="pool2", kh=2, kw=2, dh=2, dw=2)

    # 第三段
    conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256,
                      dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256,
                      dh=1, dw=1, p=p)
    conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256,
                      dh=1, dw=1, p=p)
    pool3 = mpool_op(conv3_3, name="pool3", kh=2, kw=2, dh=2, dw=2)

    # 第四段
    conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512,
                      dh=1, dw=1, p=p)
    conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512,
                      dh=1, dw=1, p=p)
    conv4_3 = conv_op (conv4_2, name="conv4_3", kh=3, kw=3, n_out=512,
                       dh=1, dw=1, p=p)
    pool4 = mpool_op (conv4_3, name="pool4", kh=2, kw=2, dh=2, dw=2)

    # 第五段
    conv5_1 = conv_op (pool4, name="conv5_1", kh=3, kw=3, n_out=512,
                       dh=1, dw=1, p=p)
    conv5_2 = conv_op (conv5_1, name="conv5_2", kh=3, kw=3, n_out=512,
                       dh=1, dw=1, p=p)
    conv5_3 = conv_op (conv5_2, name="conv5_3", kh=3, kw=3, n_out=512,
                       dh=1, dw=1, p=p)
    pool5 = mpool_op (conv5_3, name="pool5", kh=2, kw=2, dh=2, dw=2)

    # 使得结果成1维数据
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name='resh1')

    # 连接一个隐含层
    fc6 = fc_op(resh1, name="fc6", n_out=4096, p=p)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")
    fc7 = fc_op(fc6_drop, name="fc7", n_out=4096, p=p)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

    # 1000个输出节点的全连接层
    fc8 = fc_op(fc7_drop, name="fc8", n_out=1000, p=p)
    # 分类输出概率
    softmax = tf.nn.softmax(fc8)
    # 求输出概率最大的类别
    predictions = tf.argmax(softmax, 1)
    return predictions, softmax, fc8, p


def time_tensorflow_run(session, target, feed, info_string):
    # 只考虑预热轮数10轮之后的时间
    num_steps_burn_in = 10
    # 总时间
    total_duration = 0.0
    # 平方和
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run (target, feed_dict=feed)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                       (datetime.now (), i - num_steps_burn_in, duration))
                total_duration += duration
                total_duration_squared += duration * duration

    # 计算平均耗时mn 标准差sd
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec  / batch' %
           (datetime.now (), info_string, num_batches, mn, sd))


# 主函数
def run_benchmark():
    g = tf.Graph ()
    # 定义默认Graph
    with g.as_default():
        # 构造随机数据
        image_size = 224
        images = tf.Variable(tf.random_normal(
            [batch_size, image_size, image_size, 3],
            dtype=tf.float32, stddev=1e-1 ))

        keep_prob = tf.placeholder(tf.float32)
        predictions, softmax, fc8, p = inference_op(images, keep_prob)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        # 统计运行时间
        time_tensorflow_run(sess, predictions, {keep_prob: 1.0},"Forward")
        objective = tf.nn.l2_loss(fc8)
        grad = tf.gradients(objective, p)
        time_tensorflow_run(sess, grad, {keep_prob: 0.5}, "Forward-backward")


run_benchmark()