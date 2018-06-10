#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-28 19:13:22
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# Practice 返回函数
# 利用闭包返回一个计数器函数，每次调用它返回递增整数：


def createCounter():
    fs = [0]

    def counter():
        fs[0] = fs[0] + 1
        return fs[0]
    return counter


# 测试:
counterA = createCounter()
print(counterA(), counterA(), counterA(), counterA(), counterA())  # 1 2 3 4 5
counterB = createCounter()
if [counterB(), counterB(), counterB(), counterB()] == [1, 2, 3, 4]:
    print('测试通过!')
else:
    print('测试失败!')
