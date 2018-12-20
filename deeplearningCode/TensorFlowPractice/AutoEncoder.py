#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/6 20:06
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : AutoEncoder.py
# @Software: PyCharm

# 本程序实现去噪自编码器
# 此类编码器使用范围最广也最通用
# 无噪声的自编码器只需要去掉噪声，并保证隐含层节点小于输入层节点;
# Masking Noise 的自编码器只需要将高斯噪声改为随机遮挡噪声
# Variational（变分） AutoEncoder ( VAE )则相对复杂

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 定义标准的均匀分布Xaiver初始化器 fan_in是输入节点数量 fan_out是输出节点数量
# 参考：https://blog.csdn.net/nini_coded/article/details/79302820
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low,
                             maxval=high, dtype=tf.float32)


# 定义去噪自编码的class
# scale高斯噪声系数，默认为0.1
class AdditiveGaussianNoiseAuto(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        # 输入变量数
        self.n_input = n_input
        # 隐藏层节点数
        self.n_hidden = n_hidden
        # 隐藏层激活函数
        self.transfer = transfer_function
        # 高斯噪声系数，默认为0.1
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        # _initialize_weights初始化权重参数
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # 为输入数据x创建一个维度为n_input的隐藏层
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        # 提取特征的隐藏层
        self.hidden = self.transfer(tf.add(tf.matmul(
            self.x + scale * tf.random_normal((n_input, )),
            self.weights['w1']), self.weights['b1']))

        # 隐藏层之后，开始数据复原、重建，不再需要激活函数
        # 直接将隐藏层的输出self.hidden与w2相乘再加上偏置值b2
        self.reconstruction = tf.add(tf.matmul(self.hidden,
            self.weights['w2']), self.weights['b2'])

        # 使用平方误差作为损失函数，即求(最后的结果-x)^2 的平均值
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(
            self.reconstruction, self.x), 2.0))
        # 定义训练的优化器，优化损失cost
        self.optimizer = optimizer.minimize(self.cost)

        # 初始化变量
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    # 初始化权重参数函数
    def _initialize_weights(self):
        # 定义用于存放参数的字典
        all_weights = dict()
        # xavier_init函数用于初始化参数，b1 b2 w2都置为0
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,
                                                    self.n_hidden))

        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],
                                                 dtype=tf.float32))

        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,
                                                  self.n_input], dtype=tf.float32))

        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],
                                                 dtype=tf.float32))
        return all_weights

    # 定义计算损失函数cost以及训练函数，会出发一个batch数据的训练并返回cost
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict = {self.x: X, self.scale: self.training_scale})
        return cost

    # 该函数用于单独计算cost，在训练完毕后的模型性能评测上会用到，不触发训练
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x:X,
            self.scale: self.training_scale
        })

    # 返回自编码器的隐藏层输出，提供接口获取抽象特征,学习数据高阶特征
    def transform(self, X):
        return self.sess.run(self.n_hidden, feed_dict={self.x: X,
            self.scale: self.training_scale
        })

    # 接口的后半部分，将高阶特征复原为原始数据
    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction,
                             feed_dict={self.hidden: hidden})

    # 提取高阶特征；通过高阶特征复原数据（transform,generate）原数据->复原数据
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction,
            feed_dict={self.x: X, self.scale: self.training_scale
        })

    # 获取隐藏层的权重w1
    def getWeights(self):
        return self.sess.run(self.weights["w1"])

    # 获取隐藏层的偏置系数b1
    def getBiases(self):
        return self.sess.run(self.weights["b1"])


# 载入MNIST数据集用于测试模型对数据的复原效果
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 该函数用于对训练数据和测试数据标准化（数据的均值为0，标准差为1）
# 方法是：减去均值再除以标准差
def standard_scale(X_train, X_test):
    # 使用sklearn.preprocessing的StandardScaler类在训练集上fit
    # 这里为了保证模型处理数据时的一致性，训练、测试数据使用相同的Scaler
    preprocessor = prep.StandardScaler().fit(X_train)

    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

# 获取随机block数据,取0 __ len(data) - batch_size的随机整数
# 依次取batch，不放回抽取，利用效率高
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


# 使用stand_scale对训练、测试数据进行标准变换
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

# 定义总训练样本数
n_samples = int(mnist.train.num_examples)
# 最大训练轮数
training_epochs = 50
batch_size = 128
# 每隔一轮显示损失
display_step = 1

# 创建自编码器实例
autoencoder = AdditiveGaussianNoiseAuto(n_input=784,
                                        n_hidden=200,
                                        transfer_function=tf.nn.softplus,
                                        optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                        scale=0.01)

# 开始训练
for epoch in range(training_epochs):
    # 设置平均损失为0
    avg_cost = 0.
    # 总batch数
    total_batch = int(n_samples / batch_size)
    # 循环每个batch
    for i in range(total_batch):
        # 随机抽取一个block数据
        batch_xs = get_random_block_from_data(X_train, batch_size)

        # 训练batch数据，计算当前cost
        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost / n_samples

    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost = ", "{:.9f}".format(avg_cost))

print("total cost:" + str(autoencoder.calc_total_cost(X_test)))
# 下面是不同训练轮数的总损失：
# 10轮:750178.0
# 20轮:693852.0
# 30轮:634814.0
# 40轮:624854.0
# 50轮:661178.0

# 自编码器与单隐藏层的神经网络差不多，不同之处在于数据输入做了标准化，加了高斯噪声
# 其次，输出结果不是数字分类的结果，而是复原的数据，是一种无监督学习