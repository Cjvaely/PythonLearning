#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-22 19:46:15
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# Iteration迭代：如果给定一个list或tuple，
# 我们可以通过for循环来遍历这个list或tuple，这种遍历我们称为迭代
#
# dict可以迭代:
# >>> d = {'a': 1, 'b': 2, 'c': 3}
# >>> for key in d:
# ...     print(key)
#
# 迭代value，可以用for value in d.values()，
# 如果要同时迭代key和value，可以用for k, v in d.items()
#
# 请使用迭代查找一个list中最小和最大值，并返回一个tuple


def findMinAndMax(L):
    if len(L) == 0:
        return (None, None)
    elif len(L) == 1:
        return (L[0], L[0])
    else:
        return (sorted(L)[0], sorted(L)[len(L) - 1])


# 测试
if findMinAndMax([]) != (None, None):
    print('测试失败!')
elif findMinAndMax([7]) != (7, 7):
    print('测试失败!')
elif findMinAndMax([7, 1]) != (1, 7):
    print('测试失败!')
elif findMinAndMax([7, 1, 3, 9, 5]) != (1, 9):
    print('测试失败!')
else:
    print('测试成功!')
