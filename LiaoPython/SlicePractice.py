#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-22 19:13:58
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# 切片Slice练习
# 利用切片操作，实现一个trim()函数，
# 去除字符串首尾的空格，注意不要调用str的strip()方法
#


def trim(s):
    if s[:1] == ' ':
        return trim(s[1:])
    elif s[-1:] == ' ':
        return trim(s[:-1])
    else:
        return s


# 测试:
if trim('hello  ') != 'hello':
    print('测试失败!')
elif trim('  hello') != 'hello':
    print('测试失败!')
elif trim('  hello  ') != 'hello':
    print('测试失败!')
elif trim('  hello  world  ') != 'hello  world':
    print('测试失败!')
elif trim('') != '':
    print('测试失败!')
elif trim('    ') != '':
    print('测试失败!')
else:
    print('测试成功!')
