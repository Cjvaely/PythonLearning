#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-02 16:41:35
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# 错误处理
from functools import reduce


def str2num(s):
    try:
        return int(s)
    except ValueError as e:
        print('Error：', e)
        return float(s)


def calc(exp):
    ss = exp.split('+')
    ns = map(str2num, ss)
    return reduce(lambda acc, x: acc + x, ns)


def main():
    r = calc('100 + 200 + 345')
    print('100 + 200 + 345 =', r)
    r = calc('99 + 88 + 7.6')
    print('99 + 88 + 7.6 =', r)


main()
