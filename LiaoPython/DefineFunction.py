#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-22 09:08:44
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# 定义一个函数quadratic(a, b, c)，接收3个参数，返回一元二次方程
import math


def quadratic(a, b, c):
    x = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    y = (-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)
    return x, y


# 测试:
print('quadratic(2, 3, 1) =', quadratic(2, 3, 1))
print('quadratic(1, 3, -4) =', quadratic(1, 3, -4))

if quadratic(2, 3, 1) != (-0.5, -1.0):
    print('测试失败')
elif quadratic(1, 3, -4) != (1.0, -4.0):
    print('测试失败')
else:
    print('测试成功')
