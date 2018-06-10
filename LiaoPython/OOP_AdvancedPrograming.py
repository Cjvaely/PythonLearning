#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-01 21:26:43
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# Advanced Programing of OOP:面向对象高级编程
# (1)Using __slots__
# 为了达到限制的目的，Python允许在定义class的时候，定义一个特殊的__slots__变量，来限制该class实例能添加的属性：

# class Student(object):
#     __slots__ = ('name', 'age') # 用tuple定义允许绑定的属性名称
# 使用__slots__要注意，__slots__定义的属性仅对当前类实例起作用，
# 对继承的子类是不起作用的。除非在子类中也定义__slots__，
# 这样，子类实例允许定义的属性就是自身的__slots__加上父类的__slots__
#
