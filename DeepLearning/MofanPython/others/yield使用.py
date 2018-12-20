#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-25 11:31:26
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : http://www.cnblogs.com/cjvae/
# @Version : $Id$


# # 普通的函数实现
# def fib(max):
#     n, a, b = 0, 0, 1
#     while n < max:
#         print(b)
#         a, b = b, a + b
#         n = n + 1
#     return 'done'
# """不足：
# 在 fab 函数中用 print 打印数字会导致该函数可复用性较差
# 因为其他函数无法获得该函数生成的数列。
# """


# # 返回一个 List
# def fib(max):
#     n, a, b = 0, 0, 1
#     L = []
#     while n < max:
#         L.append(b)
#         a, b = b, a + b
#         n = n + 1
#     return L
# """不足
# 该函数在运行中占用的内存会随着参数 max 的增大而增大，如果要控制内存占用，最好不要用 List
# 来保存中间结果，而是通过iterable对象来迭代。
# """


# 使用yield
def fab(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        # print b
        a, b = b, a + b
        n = n + 1


# 运行
for n in fab(5):
    print(n)

# 结果
"""
带有 yield 的函数不再是一个普通函数，Python 解释器会将其视为一个 generator，
调用 fab(5) 不会执行 fab 函数，而是返回一个 iterable对象
"""
