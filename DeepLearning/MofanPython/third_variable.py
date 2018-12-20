#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/2 15:20
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : third_variable.py
# @Software: PyCharm
"""
my third deeplearning code

tensorflow中定义了某字符串是变量，它才是变量，这一点是与 Python 所不同的。

重要的点：1、tensorflow中一旦定义了变量，一定要激活变量：init = tf.global_variables_initializer()
        2、创建session(tensorflow中用来执行语句)，用于激活变量：session.run(init)
        3、通过session执行赋值语句：sess.run(update)
        4、通过session输出变量的值：sess.run(state)
"""
from __future__ import print_function
import tensorflow as tf

# 定义一个变量state，初始值是0，名字是counter
state = tf.Variable(0, name='counter')

# 定义一个常量1
one = tf.constant(1)

# 新值是变量加常量，还是一个变量
new_value = tf.add(state, one)
# 把new_value加载到state上，于是state=new_value
update = tf.assign(state, new_value)

# 定义变量之后非常重要的一步：激活所有变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 初始化所有变量
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))


"""
tensorflow: 1.1.0
"""
var = tf.Variable(0)

add_operation = tf.add(var, 1)
update_operation = tf.assign(var, add_operation)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(3):
        sess.run(update_operation)
        print(sess.run(var))
