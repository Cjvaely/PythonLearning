#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/5 10:34
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : sixth_use_activation_fun.py
# @Software: PyCharm

"""
构造典型的三层神经网络
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 添加神经网络层
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weight = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # 输入值与Weight相乘加上biases
    Wx_plus_b = tf.matmul(inputs, Weight) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
"""
得到一组类似的值：300个值，范围是-1——1
array([[-1.        ],
       [-0.99331104],
       [-0.98662207],
       [-0.97993311],
       [-0.97324415],
       [-0.96655518],
       [-0.95986622],
       [-0.95317726],
       [-0.94648829],
       [-0.93979933],
       [-0.93311037],
"""

# 给一元二次方程制造噪点，使得数据更加真实,数据范围是平均值是0，方差是0.05,与x_data一样，300个数
noise = np.random.normal(0, 0.05, x_data.shape)

y_data = np.square(x_data) - 0.5 + noise
# plt.scatter(x_data, y_data)
# plt.show()

# 暂存输入数据
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 添加隐藏层,输出需要激励函数激励
# 输入层、输出层是定好的，输入多少个数据，就是有多少个神经元，都是1层
# 隐藏层神经元个数是10个，
h1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# 添加输出层，输入数据是隐藏层的输出数据
prediction = add_layer(h1, 10, 1, activation_function=None)

# 误差的方差， reduction_indices表示函数的处理维度
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                                    reduction_indices=[1]))

# 开始训练优化，采用梯度下降法，学习效率0.1
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 执行动作必须的声明， 非常重要
sess = tf.Session()

# 旧版本和新版本的初始化所有变量
# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess.run(init)

# 可视化
fig = plt.figure()
ax = fig.add_subplot(111)
# 真实数据散点图显示
ax.scatter(x_data, y_data)
plt.ion()   # 用于连续显示
plt.show()  # 如果不用ion，每次show之后会暂停

for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # 把prediction的值用曲线显示出来
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        # 打印出误差变化
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        # 可视化误差变化数据
        plt.pause(1)
