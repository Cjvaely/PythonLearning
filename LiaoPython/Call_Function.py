#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-22 08:56:53
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# 函数调用练习 利用Python内置的hex()函数把一个整数转换成十六进制表示的字符串
n1 = 255
n2 = 1000
n1_16 = hex(n1)
n2_16 = hex(n2)
print('%d的十六进制是%s' % (n1, hex(n1)))
print('%d的十六进制是%s' % (n2, hex(n2)))
