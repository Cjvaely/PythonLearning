#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-22 21:39:22
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# 在Python中，这种一边循环一边计算的机制，称为生成器：generator
#
# 创建generator 方法一,把一个列表生成式的[]改成()，就创建了一个generator：
#
# >>> g = (x * x for x in range(10))
# >>> g
# <generator object <genexpr> at 0x1022ef630>
# 通过next()函数获得generator的下一个返回值
# >>> next(g)
# 0
# >>> next(g)
# 1
# >>> next(g)
#
# 正确的方法是使用for循环，因为generator也是可迭代对象：
# >>> for n in g:
# ... 	print(n)
# ...
# 0
# 1
# 4
# 9
# ...
# 斐波拉契数列,输出前n个数:
# def fib(max):
# 	n, a, b = 0, 0, 1
# 	while n < max:
#   	print(b)
#   	a, b = b, a + b
#   	n = n + 1
# 	return 'done'
#
# 上面的函数和generator仅一步之遥。
# 要把fib函数变成generator，
# 只需要把print(b)改为yield b就可以了：
#
# 创建generator 方法二:
# def fib(max):
# n, a, b = 0, 0, 1
# while n < max:
# 	yield b
#   a, b = b, a + b
#   n = n + 1
# return 'done'
#
# 如果一个函数定义中包含yield关键字，
# 那么这个函数就不再是一个普通函数，
# 而是一个generator
