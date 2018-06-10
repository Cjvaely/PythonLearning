#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-21 20:39:31
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

print('My name is:', chr(38472), chr(38742))
name = input('Input your name:')
print(name)
# 要注意区分'ABC'和b'ABC'，前者是str，
# 后者虽然内容显示得和前者一样，但bytes的每个字符都只占用一个字节
# Python的字符串类型是str，在内存中以Unicode表示，一个字符对应若干个字节。
# 如果要在网络上传输，或者保存到磁盘上，就需要把str变为以字节为单位的bytes

# >>> 'ABC'.encode('ascii')
#b'ABC' 

# 在Python中，采用的格式化方式和C语言是一致的，用%实现，举例如下：
# >>> 'Hello, %s' % 'world'
# 'Hello, world'
# >>> 'Hi, %s, you have $%d.' % ('Michael', 1000000)
# 'Hi, Michael, you have $1000000.'

print('%2d-%02d' % (3, 1))
print('%.2f' % 3.1415926)