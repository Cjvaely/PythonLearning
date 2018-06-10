#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-08 14:09:04
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# itertools:提供了非常有用的用于操作迭代对象的函数
# 无限"迭代器":
# 练习：计算圆周率可以根据公式：
# 利用Python提供的itertools模块，我们来计算这个序列的前N项和
import itertools
from functools import reduce


def pi(N):
    ' 计算pi的值 '
    # step 1: 创建一个奇数序列: 1, 3, 5, 7, 9, ...
    n1 = itertools.count(1, 2)
    # step 2: 取该序列的前N项: 1, 3, 5, 7, 9, ..., 2*N-1.
    n2 = itertools.takewhile(lambda x: x <= 2 * N - 1, n1)
    # step 3: 添加正负符号并用4除: 4/1, -4/3, 4/5, -4/7, 4/9, ...
    n3 = map(lambda x: 4 / x if x % 4 == 1 else -4 / x, n2)
    # step 4: 求和:
    sum = reduce(lambda x, y: x + y, n3)
    return sum


# 测试:
print(pi(10))
print(pi(100))
print(pi(1000))
print(pi(10000))
assert 3.04 < pi(10) < 3.05
assert 3.13 < pi(100) < 3.14
assert 3.140 < pi(1000) < 3.141
assert 3.1414 < pi(10000) < 3.1415
print('ok')
