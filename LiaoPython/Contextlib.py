#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-08 15:20:01
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# contextlib:
# with open('/path/to/file', 'r') as f:
#     f.read()

from contextlib import contextmanager


# class Query(object):

#     def __init__(self, name):
#         self.name = name

#     def query(self):
#         print('Query info about %s...' % self.name)


# @contextmanager
# def create_query(name):
#     print('Begin')
#     q = Query(name)
#     yield q
#     print('End')


# with create_query('Bob') as q:
#     q.query()

# 很多时候，我们希望在某段代码执行前后自动执行特定代码，可以也。用@contextmanager实现例如：
@contextmanager
def tag(name):
    print("<%s>" % name)
    yield
    print("</%s>" % name)


with tag("h1"):
    print("hello")
    print("world")
# 代码的执行顺序是：
# with首先语句执行yield之前的语句，因此打印出<h1>;
# yield会调用执行with语句内部的所有语句，因此出打印hello状语从句：world;
# 执行求最后yield之后的语句，打印出</h1>
