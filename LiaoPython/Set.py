#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-22 01:47:02
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# set和dict类似，也是一组key的集合，但不存储value。
# 由于key不能重复，所以，在set中，没有重复的key。
# 要创建一个set，需要提供一个list作为输入集合：

# >>> s = set([1, 2, 3])
# >>> s
# {1, 2, 3}
# 注意，传入的参数[1, 2, 3]是一个list，而显示的{1, 2, 3}
# 只是告诉你这个set内部有1，2，3这3个元素，显示的顺序也不表示set是有序的
# 通过add(key)方法可以添加元素到set中，可以重复添加，但不会有效果
# >>> s.add(4)
# >>> s
# {1, 2, 3, 4}
# 通过remove(key)方法可以删除元素
# >>> s.remove(4)
# >>> s
# {1, 2, 3}

# set可以看成数学意义上的无序和无重复元素的集合，因此，两个set可以做数学意义上的交集、并集等操作：
# >>> s1 = set([1, 2, 3])
# >>> s2 = set([2, 3, 4])
# >>> s1 & s2
# {2, 3}
# >>> s1 | s2
# {1, 2, 3, 4}
#
# set的原理和dict一样，所以，同样不可以放入可变对象，
# 因为无法判断两个可变对象是否相等，也就无法保证set内部“不会有重复元素”
#
