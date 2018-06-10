#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-21 21:52:09
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# 利用循环依次对list中的每个名字打印出Hello, xxx!
print('Use for in------------------')
L = ['Bart', 'Lisa', 'Adam']
for x in L:
    print('Hello,', x, '!')
print('Use while version1--------------------')
n = 0
while n < len(L):
    print('Hello,%s' % L[n], '!')
    n = n + 1
print('Use while version2--------------------')
n = len(L)
while n > 0:
    print('Hello,%s' % L[-n], '!')
    n = n - 1
