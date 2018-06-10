#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-07 19:14:59
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# collections:内建的一个集合模块，提供了许多有用的集合类。

# namedtuple:一个函数，它用来创建一个自定义的tuple对象，并且规定了tuple元素的个数，
# 并可以用属性而不是索引来引用tuple的某个元素
# namedtuple('名称', [属性list])

# deque:高效实现插入和删除操作的双向列表，适合用于队列和栈
# q = deque(['a', 'b', 'c'])
# q.append/pop('x')
# q.append/popleft('y')

# defaultdict:使用dict时,引用的Key不存在,返回一个默认值
# dd = defaultdict(lambda: 'N/A')

# base64编码：请写一个能处理去掉=的base64解码函数：
import base64


def safe_base64_decode(s):
    if len(s) % 4 != 0:
        t = s + b'=='
        return base64.b64decode(t)
    else:
        return base64.b64decode(s)


# 测试:
assert b'abcd' == safe_base64_decode(
    b'YWJjZA=='), safe_base64_decode('YWJjZA==')
assert b'abcd' == safe_base64_decode(b'YWJjZA'), safe_base64_decode('YWJjZA')
print('ok')
