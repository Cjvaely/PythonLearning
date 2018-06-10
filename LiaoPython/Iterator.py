#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-23 19:42:21
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# 迭代对象和迭代器的关系
# 可以直接作用于for循环的对象统称为可迭代对象：Iterable
#
# >>> from collections import Iterable
# >>> isinstance([], Iterable)
# True

# 可以被next()函数调用并不断返回下一个值的对象称为迭代器：Iterator
# 可以使用isinstance()判断一个对象是否是Iterator对象：

# >>> from collections import Iterator
# >>> isinstance((x for x in range(10)), Iterator)
# True
# >>> isinstance([], Iterator)
# False

# 把list、dict、str等Iterable变成Iterator可以使用iter()函数：
# >>> isinstance(iter([]), Iterator)
# True
# >>> isinstance(iter('abc'), Iterator)
# True

# 小结
# 凡是可作用于for循环的对象都是Iterable类型；

# 凡是可作用于next()函数的对象都是Iterator类型，它们表示一个惰性计算的序列；

# 集合数据类型如list、dict、str等是Iterable但不是Iterator，
# 不过可以通过iter()函数获得一个Iterator对象。
# Python的for循环本质上就是通过不断调用next()函数实现的，例如：
# for x in [1, 2, 3, 4, 5]:
#     pass
# 实际上完全等价于：

# #首先获得Iterator对象:
# it = iter([1, 2, 3, 4, 5])
# # 循环:
# while True:
#     try:
#         # 获得下一个值:
#         x = next(it)
#     except StopIteration:
#         # 遇到StopIteration就退出循环
#         break
