#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-01 20:12:31
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# 获取对象的消息:
#
# 拿到一个对象的引用时，
# 如何知道这个对象是什么类型、有哪些方法呢？
#
# 判断对象类型，使用type()函数：
# 基本类型都可以用type()判断.
#
# >>> type(123)
# <class 'int'>
# >>> type('str')
# <class 'str'>
#
# 如果一个变量指向函数或者类，也可以用type()判断：
# >>> type(abs)
# <class 'builtin_function_or_method'>
# >>> type(a)
# <class '__main__.Animal'>
#
# type()函数返回的是Class类型
# >>> type(123)==type(456)
# True
# >>> type(123)==int
# True
# >>> type('abc')==type('123')
# True
# >>> type('abc')==str
# True
# >>> type('abc')==type(123)
# False
#
# 还有isinstance()
# 能用type()判断的基本类型也可以用isinstance()判断
#
# 并且还可以判断一个变量是否是某些类型中的一种，
# 比如下面的代码就可以判断是否是list或者tuple：
# >> > isinstance([1, 2, 3], (list, tuple))
# True
# >> > isinstance((1, 2, 3), (list, tuple))
# True
#
# 优先使用isinstance()判断类型，可以将指定类型及其子类“一网打尽”。

# 使用dir():获得一个对象的所有属性和方法，可以使用dir()函数
# >>> dir('ABC')
# ['__add__', '__class__',..., '__subclasshook__',
#  'capitalize', 'casefold',..., 'zfill']
#
# 我们自己写的类，如果也想用len(myObj)的话，就自己写一个__len__()方法：
# >>> class MyDog(object):
# ...     def __len__(self):
# ...         return 100
# ...
# >>> dog = MyDog()
# >>> len(dog)
# 100
#
#
