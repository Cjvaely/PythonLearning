#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-21 22:25:02
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# break in circulation
# break 提前退出
n = 1
while n <= 100:
    print(n)
    n = n + 1
    if n > 10:
        break
print('END')
