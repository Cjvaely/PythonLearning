#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-24 01:38:39
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# 内建函数之filter()：用于过滤序列,接收一个函数和一个序列。
# 和map()不同的是，filter()把传入的函数依次作用于每个元素，
# 然后根据返回值是True还是False决定保留还是丢弃该元素

# 用filter求素数，埃氏筛法


def _odd_iter():  # 从3开始构造奇数序列
    n = 1
    while True:
        n = n + 2
        yield n


def _not_divisible(n):  # 定义一个筛选函数, 筛选掉非素数
    return lambda x: x % n > 0


def primes():  # 最后，定义一个生成器，不断返回下一个素数
    yield 2
    it = _odd_iter()  # 初始序列
    while True:
        n = next(it)  # 返回序列的第一个数
        yield n
        it = filter(_not_divisible(n), it)  # 构造新序列
