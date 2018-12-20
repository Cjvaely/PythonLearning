#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-10-19 00:22:38
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : http://www.cnblogs.com/cjvae/
# @Version : $Id$

import tensorflow as tf

g1 = tf.Graph()

with g1.as_default():
	# 在计算图g1中定义变量'v'，并设置初始值为0
	v = tf.get_variable(
		"v", initializer = tf.zeros_initializer())

g2 = tf.Graph()
with g2.as_default():
	# 在计算图g2中定义变量'v'，并设置初始值为1
	v = tf.get_variable(
		"v", initializer = tf.ones_initializer())

# 在计算图g1中读取变量“v”的取值
with tf.Session(graph=g1) as sess:
	tf.initializer_all_variables().run()
	with tf.variable_scope("", reuse=True):
		# 在计算图g1中，变量"v"的取值应该是0，所以下面输出为[0.]
		print(sess.run(tf.get_variable("v")))

# 在计算图g2中读取变量“v”的取值
with tf.Session(graph=g2) as sess:
	tf.initializer_all_variables().run()
	with tf.variable_scope("", reuse=True):
		# 在计算图g1中，变量"v"的取值应该是1，所以下面输出为[1.]
		print(sess.run(tf.get_variable("v")))