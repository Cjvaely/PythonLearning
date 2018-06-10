#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-22 17:38:16
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# 构造一个1, 3, 5, 7, ..., 99的列表,可以通过循环实现
L = []
n = 1
while n <= 99:
    L.append(n)
    n = n + 2
print(L)
