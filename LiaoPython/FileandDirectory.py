#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-05 20:16:30
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# Operate file and directory

# 查看当前目录的绝对路径:
# os.path.abspath('.')

# 在某个目录下创建一个新目录，首先把新目录的完整路径表示出来:
# os.path.join('/Users/michael', 'testdir')

# 练习：1.利用os模块编写一个能实现dir -l输出的程序。
import os
import re
print('Work1---------------------------------------')
path = input('Input the path:')
dirl = [x for x in os.listdir(path) if re.match(
    '^[a-z]+$', os.path.splitext(x)[0])]
print(dirl)

# 2.编写一个程序，能在当前目录以及当前目录的所有子目录下
# 查找文件名包含指定字符串的文件，
# 并打印出相对路径。
print('Work2---------------------------------------')
s = input('Input the string you want find:')
ls = []
for x in os.listdir('.'):
    if s in x:
        ls.append(x)
        print(x, "\n")
    if os.path.isdir(x):
        for y in os.listdir('./' + x):
            if s in y:
                ls.append('./' + y)
                print('./' + y + "\n")
print(ls)
