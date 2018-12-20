#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/22 14:43
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : MNIST_Train.py
# @Software: PyCharm
# MNIST 实现完整功能
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST 数据集相关的常数
input_node = 784    # 输入层的节点数，对于MNIST就是图片像素数
output_node = 10    # 输出层的节点数，对于MNIST就是类别数目0-9

# 配置神经网络的参数
layer1_node = 500   # 隐藏层节点数500，这里使用只有一个的隐藏层的网络结构
batch_size = 100    # 一个训练batch中训练数据的个数
                    # 数字越小，训练过程越接近的随机梯度下降；越大接近梯度下降
learning_rate_base = 0.8    # 基础学习率
learning_rate_decay = 0.99  # 学习率的衰减率

regularization_rate = 0.0001    # 描述模型复杂度的正则化项在损失函数中的系数
training_steps = 200000          # 训练轮数
moving_average_decay = 0.99     # 滑动平均衰减率

# 给定神经网络的输入和所有参数，计算前向传播结果
# 此处定义了激活函数ReLU的三层全连接神经网络
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 当没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class == None:
        # 计算隐藏层的前向传播结果，这里使用了ReLU激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1, ) + biases1)
        # 计算输出层的前向传播结果，损失函数一并计算softmax函数
        # 这里不需要加入激活函数，而且不加softmax不会影响预测结果
        return tf.matmul(layer1, weights2) + biases2

    else:
        # 首先，使用avg_class.average函数来计算得出的变量的滑动平均值
        # 然后计算相应的前向传播结果
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1)
        )
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

# 训练模型的过程
def train(mnist):
    # 真实数据
    x = tf.placeholder(tf.float32, [None, input_node], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, output_node], name='y-input')

    # 生成隐藏层的参数
    weights1 = tf.Variable(
        tf.truncated_normal([input_node, layer1_node], stddev=0.1)
    )
    biases1 = tf.Variable(tf.constant(0.1, shape=[layer1_node]))
    # 生成输出层的参数
    weights2 = tf.Variable(
        tf.truncated_normal([layer1_node, output_node], stddev=0.1)
    )
    biases2 = tf.Variable(tf.constant(0.1, shape=[output_node]))

    # 计算当前参数下神经网络前向传播的结果，这里avg_class=None
    # 所以函数不会用参数的滑动平均值
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义训练轮数的变量，该变量不需计算初始化滑动平均值
    # 所以该变量为不可训练的变量（trainable=False）。
    # 在使用TensorFlow训练神经网络，一般会将代表训练轮数的变量指定为不可训练的
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    # 给定训练轮数的变量可以加快训练早期变量的更新速度
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step
    )

    # 在所有代表神经网络参数的变量上使用滑动平均。
    # 辅助变量如global_step就不需要了
    variable_averages_op = variable_averages.apply(
        tf.trainable_variables()
    )
    # 计算使用了滑动平均的前向传播结果。滑动平均值不会改变变量本身取值，
    # 但是会维护一个影子变量记录滑动平均值
    average_y = inference(
        x, variable_averages, weights1, biases1, weights2, biases2
    )

    # 这里使用的是sparse_softmax_cross_entropy_with_logits函数计算交叉熵
    # 因为标准答案是一个长度为10的一维数组，该函数需要的是一个正确答案的数字
    # 因此，这里需要使用tf.argmax来获取正确答案对应的类别编号
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1)
    )
    # 计算当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    # 计算模型的正则化损失，一般只计算神经网络边上权重的正则化损失，不使用偏置项
    regularization = regularizer(weights1) + regularizer(weights2)
    # 总损失等于交叉熵的损失和正则化损失的和
    loss = cross_entropy_mean + regularization
    # 设置指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        learning_rate_base,     # 基础的学习率，随着迭代进行，更新变量时使用
                                # 学习率在这个基础上递减
        global_step,            # 当前的迭代轮数
        mnist.train.num_examples / batch_size,  # 过完所有训练数据需要的迭代次数
        learning_rate_decay     # 学习率衰减速度
    )
    # 使用tf.train.GradientDescentOptimizer优化算法来优化损失函数
    # 这里损失函数包括了交叉熵损失和L2正则化损失
    train_steps = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 在训练神经网络模型时，每过一遍数据需要同时反向传播更新神经网络参数
    # 以及每一个参数的滑动平均值
    with tf.control_dependencies([train_steps, variable_averages_op]):
        train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 准备验证数据，一般在神经网络的训练过程中会通过验证数据判断停止条件和评判标准
        validate_feed = {
            x:mnist.validation.images,
            y_:mnist.validation.labels
        }
        # 准备测试数据，在真实的应用中，这部分数据训练时不可见，只是模型的评价标准
        test_feed = {
            x:mnist.test.images,
            y_:mnist.test.labels
        }

        # 迭代训练神经网络
        for i in range(training_steps):
            # 每1000轮输出一次测试结果
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                test_acc = sess.run(accuracy, feed_dict=test_feed)
                # 输出正确率信息
                print('After %d training step(s), validation accuracy using average model'
                      ' is %g, test accuracy using average model is %g ' % (i, validate_acc, test_acc))
                # 产生这=一轮使用的一个batch训练数据，并运行训练过程
                xs, ys = mnist.train.next_batch(batch_size)
                sess.run(train_op, feed_dict={x: xs, y_:ys})
        # 检测测试数据上模型的最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('After %d training step(s), test accuracy using average.Model is %g' % (training_steps, test_acc))


# 主程序入口
def main(argv=None):
    # 声明处理MNIST数据集的类，初始化时会自动下载数据
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    train(mnist)
    return 0


# tensorflow 提供的主程序入口
if __name__ == '__main__':
    tf.app.run()

