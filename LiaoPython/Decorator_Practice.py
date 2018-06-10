#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-28 22:11:00
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$


# decorator Practice:请设计一个decorator，它可作用于任何函数上，
# 并打印该函数的执行时间：

import time

import functools

print('Test one:')


def metric(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kw):
        print('%s executed in %s' %
              (fn.__name__, time.asctime(time.localtime(time.time()))))
        return fn(*args, **kw)
    return wrapper

# 测试


@metric
def fast(x, y):
    time.sleep(0.0012)
    return x + y


@metric
def slow(x, y, z):
    time.sleep(0.1234)
    return x * y * z


f = fast(11, 22)
s = slow(11, 22, 33)

print(f)
print(s)

if f != 33:
    print('测试失败!')
elif s != 7986:
    print('测试失败!')
else:
    print('测试成功！')


# 请编写一个decorator，
#  能在函数调用的前后打印出'begin call'和'end call'的日志。
# print('\nTest two:')


# def log(func):
#     @functools.wraps(func)
#     def fn(*args, **kw):
#         print('begin call %s():' % func.__name__)
#         call = func(*args, **kw)
#         print('end call %s()' % func.__name__)
#         return call
#     return fn


# @log
# def now():
#     print('2018-05-29')
#     return ''


# print(now())  # now = log(now())


# 写出一个@log的decorator，使它既支持：
# @log
# def f():
#     pass
# 又支持：

# @log('execute')
# def f():
#     pass
print('\nTest three:')


def log(text):
    if isinstance(text, str):
        def decorator(func):
            @functools.wraps(func)
            def w(*args, **kw):
                print('%s call %s()' % (text, func.__name__))
                return func(*args, **kw)
            return w
        return decorator
    else:
        @functools.wraps(text)
        def w(*args, **kw):
            print('call %s()' % text.__name__)
            return text(*args, **kw)
        return w
# 测试


@log
def test1():
    print('2018-05-29')
    return ''


print('Test of first:')
print(test1())


@log('excute')
def test2():
    print('2018-05-29')
    return ''


print('Test of second:')
print(test2())
