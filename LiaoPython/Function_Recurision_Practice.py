#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-22 16:22:13
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# 汉诺塔的移动可以用递归函数非常简单地实现
# 需求：打印出把所有盘子从A借助B移动到C的方法


def move(n, a, b, c):
    if n == 1:
        print(a, '-->', c)
    else:
        move(n - 1, a, c, b)
        move(1, a, b, c)
        move(n - 1, b, a, c)

        # 期待输出:
        # A --> C
        # A --> B
        # C --> B
        # A --> C
        # B --> A
        # B --> C
        # A --> C


move(3, 'A', 'B', 'C')
