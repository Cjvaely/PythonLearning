#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-21 21:21:02
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# tuple和list非常类似，但是tuple一旦初始化就不能修改，
# 比如同样是列出同学的名字：
# >>> classmates = ('Michael', 'Bob', 'Tracy')
# 现在，classmates这个tuple不能变了，它也没有append()，insert()这样的方法。
# 其他获取元素的方法和list是一样的，你可以正常地使用classmates[0]，classmates[-1]
# 但不能赋值成另外的元素。

# 最后来看一个“可变的”tuple：
# >>> t = ('a', 'b', ['A', 'B'])
# >>> t[2][0] = 'X'
# >>> t[2][1] = 'Y'
# >>> t
# ('a', 'b', ['X', 'Y'])
# 
# 表面上看，tuple的元素确实变了，但其实变的不是tuple的元素，而是list的元素。
# tuple一开始指向的list并没有改成别的list，
# 所以，tuple所谓的“不变”是说，tuple的每个元素，指向永远不变。
# 即指向'a'，就不能改成指向'b'，指向一个list，就不能改成指向其他对象，但指向的这个list本身是可变的！