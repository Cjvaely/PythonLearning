#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-24 17:03:00
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# 切片实例：
L = ['Michael', 'Sarah', 'Tracy', 'Bob', 'Jack']

print('L[0:3] =', L[0:3])
print('L[:3] =', L[:3])
print('L[1:3] =', L[1:3])
print('L[-2:] =', L[-2:])

R = list(range(100))
print('R[:10] =', R[:10])
print('R[-10:] =', R[-10:])
print('R[10:20] =', R[10:20])
print('R[:10:2] =', R[:10:2])
print('R[::5] =', R[::5])
