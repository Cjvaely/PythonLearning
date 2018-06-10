#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-28 19:42:42
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# test


def counter():
    def f(j):
        def g():
            return j * j
        return g
    fs = []
    for i in range(1, 4):
        fs.append(f(i))
    return fs


f1, f2, f3 = counter()
print(f1())
print(f2())
print(f3())
