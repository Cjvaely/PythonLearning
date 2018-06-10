#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-24 17:19:58
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# Sorted 排序算法
# Python内置的sorted()函数就可以对list进行排序：
# >>> sorted([36, 5, -12, 9, -21])
# [-21, -12, 5, 9, 36]

# sorted()函数也是一个高阶函数，它还可以接收一个key函数来实现自定义的排序，例如按绝对值大小排序：

# >>> sorted([36, 5, -12, 9, -21], key=abs)
# [5, 9, -12, -21, 36]
# key指定的函数将作用于list的每一个元素上，并根据key函数返回的结果进行排序

# 再看一个字符串排序的例子：
# >>> sorted(['bob', 'about', 'Zoo', 'Credit'])
# ['Credit', 'Zoo', 'about', 'bob']
#
# >>> sorted(['bob', 'about', 'Zoo', 'Credit'], key=str.lower)
# ['about', 'bob', 'Credit', 'Zoo']

# 反向排序：
# 要进行反向排序，不必改动key函数，可以传入第三个参数reverse=True：
# >>> sorted(['bob', 'about', 'Zoo', 'Credit'], key=str.lower, reverse=True)
# ['Zoo', 'Credit', 'bob', 'about']

# 练习: 用一组tuple表示学生名字和成绩：
# L = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]
# 请用sorted()对上述列表分别按名字排序：
L = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]


def by_name(t):
    name = t[0]
    return name


L2 = sorted(L, key=by_name)
print(L2)

# 请用sorted()对上述列表分别按名字排序：


def by_score(t):
    score = t[1]
    return score


L2 = sorted(L, key=by_score)
print(L2)
