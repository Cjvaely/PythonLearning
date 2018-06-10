#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-05 17:07:59
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# document test
# 练习: 函数对fact(n)关系编写文档测试并执行：
#


def fact(n):
    '''
    Calculate 1*2*...*n

    >>> fact(1)
    1
    >>> fact(10)
    3628800
    >>> fact(-1)
    Traceback (most recent call last):
        ...
    ValueError
    '''
    if n < 1:
        raise ValueError()
    if n == 1:
        return 1
    return n * fact(n - 1)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
