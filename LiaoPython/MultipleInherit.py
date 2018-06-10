#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-02 14:59:06
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# 多重继承：
# class Dog(Animal, Runnable):
# 	pass
# 通过多重继承，一个子类就可以同时获得多个父类的所有功能。

# MixIn的目的就是给一个类增加多个功能，这样，在设计类的时候，
# 我们优先考虑通过多重继承来组合多个MixIn的功能，而不是设计多层次的复杂的继承关系
#
# 只允许单一继承的语言（如Java）不能使用MixIn的设计
