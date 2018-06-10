#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-21 21:41:56
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# for...in...循环
names = ['Michael', 'Bob', 'Tracy']
for name in names:
	print(name)

# 求和
sum = 0
for x in (1,2,3,4,5,6,7,8,9,10):
	sum = sum + x
print(sum)

# 求0-100和， range(5)——>表示0,1,2,3,4
sum = 0
for x in range(101):
	sum = sum + x
print(sum)
