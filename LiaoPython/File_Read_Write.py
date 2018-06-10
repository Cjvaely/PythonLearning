#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-05 17:28:37
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# File Read and Write
# 读写文件就是请求操作系统打开一个文件对象（通常称为文件描述符），
# 然后，通过操作系统提供的接口从这个文件对象中读取数据（读文件），
# 或者把数据写入这个文件对象（写文件）。

# 读文件： open('/Users/michael/test.txt', 'r')

# 练习：请将本地一个文本文件读为一个str并打印出来：
fpath = r'C:\Windows\system.ini'

with open(fpath, 'r') as f:
    s = f.read()
    print(str(s))
