#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-28 20:44:28
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# 匿名函数lambda:不需要显式地定义函数，直接传入匿名函数更方便。

# 匿名函数lambda x: x * x实际上就是：
# def f(x):
#     return x * x
# 关键字lambda表示匿名函数，冒号前面的x表示函数参数。
# 有个限制，就是只能有一个表达式，不用写return，
# 返回值就是该表达式的结果
# 因为函数没有名字，不必担心函数名冲突
#
# 也可以把匿名函数赋值给一个变量，再利用变量来调用该函数：
# >>> f = lambda x: x * x
# >>> f
# <function <lambda> at 0x101c6ef28>
# >>> f(5)
# 25

# 也可以把匿名函数作为返回值返回，比如：
# def build(x, y):
#     return lambda: x * x + y * y
# build(1, 3)()
# 10

# 练习 请用匿名函数改造下面的代码：


# def is_odd(n):
#     return n % 2 == 1


# L = list(filter(is_odd, range(1, 20)))
# print(L)

L = list(filter(lambda x: x % 2 == 1, range(1, 20)))
print(L)

# print(list(filter(lambda x: x % 2 == 1, range(1, 20))))
