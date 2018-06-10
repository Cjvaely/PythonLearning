#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-22 16:01:10
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# 递归函数：在函数内部，可以调用其他函数。如果一个函数在内部调用自身本身，这个函数就是递归函数。

# def fact(n):
#     if n==1:
#         return 1
#     return n * fact(n - 1)
#     容易堆栈溢出
#
# 改进: 尾递归优化
# def fact(n):
#     return fact_iter(n, 1)

# def fact_iter(num, product):
#     if num == 1:
#         return product
#     return fact_iter(num - 1, num * product)
