#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-21 21:14:33
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# >>> classmates = ['Michael', 'Bob', 'Tracy']
# >>> classmates
# ['Michael', 'Bob', 'Tracy']
# 变量classmates就是一个list。用len()函数可以获得list元素的个数

# >>> classmates[0]
# 'Michael'
# >>> classmates[-1]
# 'Tracy'

# list是一个可变的有序表，所以，可以往list中追加元素到末尾
# >>> classmates.append('Adam')
# >>> classmates
# ['Michael', 'Bob', 'Tracy', 'Adam']

# 要删除list末尾的元素，用pop()方法：
# >>> classmates.pop()
# 'Adam'
# >>> classmates
# ['Michael', 'Jack', 'Bob', 'Tracy']

#要把某个元素替换成别的元素，可以直接赋值给对应的索引位置：
# >>> classmates[1] = 'Sarah'
# >>> classmates
# ['Michael', 'Sarah', 'Tracy'] 