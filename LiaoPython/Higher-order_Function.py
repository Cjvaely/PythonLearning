#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-23 19:55:48
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# Higher-order function 高阶函数：

# 变量可以指向函数, 即：变量可以指向函数
# >>> abs
# <built-in function abs>
#
# >>> f = abs
# >>> f
# <built-in function abs>
#
# 此时变量具有函数的作用：
# >>> f = abs
# >>> f(-10)
# 10

# 函数名也是变量: 函数名其实就是指向函数的变量
#
# 如果把abs指向其他对象，会有什么情况发生？
# >>> abs = 10
# >>> abs(-10)
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# TypeError: 'int' object is not callable
# 要恢复abs函数，请重启Python交互环境

# 传入参数: 一个函数就可以接收另一个函数作为参数，
# 这种函数就称之为高阶函数
# 一个最简单的高阶函数：
# def add(x, y, f):
#     return f(x) + f(y)
# 实例:


def add(x, y, f):
    return f(x) + f(y)


print(add(-5, 6, abs))
