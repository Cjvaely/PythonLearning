#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-21 22:28:43
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# continue
# 遇到continue 先结束当前循环
n = 0
while n < 10:
    n = n + 1
    if n % 2 == 0:
        continue
    print(n)
