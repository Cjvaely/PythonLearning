#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-22 08:47:05
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# 再议不可变对象
# 对于可变对象，比如list，对list进行操作，list内部的内容是会变化的，比如：
# >>> a = ['c', 'b', 'a']
# >>> a.sort()
# >>> a
# ['a', 'b', 'c']
#
# 而对于不可变对象，比如str，对str进行操作呢：
# >>> a = 'abc'
# >>> a.replace('a', 'A')
# 'Abc'
# >>> a
# 'abc'
#
# 要始终牢记的是，a是变量，而'abc'才是字符串对象！
# 有些时候，我们经常说，对象a的内容是'abc'，
# 但其实是指，a本身是一个变量，它指向的对象的内容才是'abc'
# 当我们调用a.replace('a', 'A')时，实际上调用方法replace是作用在字符串对象'abc'上的，
# 而这个方法虽然名字叫replace，但却没有改变字符串'abc'的内容。
# 相反，replace方法创建了一个新字符串'Abc'并返回，
# 如果我们用变量b指向该新字符串，就容易理解了，变量a仍指向原有的字符串'abc'，
# 但变量b却指向新字符串'Abc'了
#
