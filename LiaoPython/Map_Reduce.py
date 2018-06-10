#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-23 20:06:26
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# map/reduce
# map(): 接收两个参数，一个是函数，一个是Iterable，
# map将传入的函数依次作用到序列的每个元素，
# 并把结果作为新的Iterator返回
#
# 举例说明，比如我们有一个函数f(x)=x2，
# 要把这个函数作用在一个
# list [1, 2, 3, 4, 5, 6, 7, 8, 9]上，
# 就可以用map()实现如下：
#
# >>> def f(x):
# ...     return x * x
# ...
# >>> r = map(f, [1, 2, 3, 4, 5, 6, 7, 8, 9])
# >>> list(r)
# [1, 4, 9, 16, 25, 36, 49, 64, 81]

# reduce(): 把一个函数作用在一个序列
# [x1, x2, x3, ...]上，这个函数必须接收两个参数，
# reduce把结果继续和序列的下一个元素做累积计算，
# 其效果就是：
# reduce(f, [x1, x2, x3, x4]) = f(f(f(x1, x2), x3), x4)
#
# 比方说对一个序列求和，就可以用reduce实现：
# >>> from functools import reduce
# >>> def add(x, y):
# ...     return x + y
# ...
# >>> reduce(add, [1, 3, 5, 7, 9])
# 25
#
# 写出把str转换为int的函数：
# >>> from functools import reduce
# >>> def fn(x, y):
# ...     return x * 10 + y
# ...
# >>> def char2num(s):
# ...     digits = {'0': 0, '1': 1, '2': 2, '3':
#  3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
# ...     return digits[s]
# ...
# >>> reduce(fn, map(char2num, '13579'))
# 13579


# 练习1: 利用map()函数，把用户输入的不规范的英文名字，
# 变为首字母大写，其他小写的规范名字。
# 输入：['adam', 'LISA','barT']，
# 输出：['Adam', 'Lisa', 'Bart']：

# 有错误
from functools import reduce


def normalize(name):
    n = 1
    name = name.replace(name[:1], name[:1].upper())[:]
    while n < len(name):
        name = name.replace(name[n], name[n].lower())[:]
        n += 1
    return name


# 借鉴:
# def normalize(name):
#     return name[0].upper() + name[1:].lower()


# 测试:
L1 = ['adam', 'LISA', 'barT']
L2 = list(map(normalize, L1))
# s = normalize('adam')
print(L2)
# print(s)


# 练习2 编写一个prod()函数，可以接受一个list并利用reduce()求积
def prod(L):
    def multiply(x, y):
        return x * y
    return reduce(multiply, L)


print('3 * 5 * 7 * 9 =', prod([3, 5, 7, 9]))
if prod([3, 5, 7, 9]) == 945:
    print('测试成功!')
else:
    print('测试失败!')


# 练习3 利用map和reduce编写一个str2float函数，把字符串'123.456'转换成浮点数123.456：
DIGITS = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
          '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}


# 自己写的，不太对，有待改进
# def str2float(s):
#     def fn(x, y):
#         return x * 10 + y

#     def char2num(s):
#         return DIGITS[s]
# return reduce(fn, map(char2num, s[:3])) + reduce(fn, map(char2num,
# s[-3:])) / 1000


# 借鉴：更加精炼
def str2float(s):
    def fn(x, y):
        return x * 10 + y

    def char2num(s):
        return DIGITS[s]
    str = s.split('.')
    # str[0] = ['123'], str[1] = '456'
    return reduce(fn, map(char2num, str[0])) + reduce(fn, map(char2num, str[1])) / (10 ** len(str[1]))


print('str2float(\'123.456\') =', str2float('123.456'))
if abs(str2float('123.456') - 123.456) < 0.00001:
    print('测试成功!')
else:
    print('测试失败!')
