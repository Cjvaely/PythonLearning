#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-07 11:27:47
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$


# 正则表达式
# 精确匹配。用\d可以匹配一个数字，\w可以匹配一个字母或数字
# .可以匹配任意字符
# *表示任意个字符（包括0个）
# +表示至少一个字符
# 用?表示0个或1个字符
# 用{n}表示n个字符
# 用{n,m}表示n-m个字符
#
# eg:\d{3}\s+\d{3,8}
# \d{3}		表示匹配三个数字
# \s+		表示匹配至少一个空格
# \d{3,8}	表示匹配3-8个数字

# 进阶：[]表示范围，比如：
# [0-9a-zA-Z\_]可以匹配一个数字、字母或者下划线；
# [0-9a-zA-Z\_]+可以匹配至少由一个数字、字母或者下划线组成的字符串
'''
[a-zA-Z\_][0-9a-zA-Z\_]*可以匹配由字母或下划线开头，
后接任意个由一个数字、字母或者下划线组成的字符串，
也就是Python合法的变量

[a-zA-Z\_][0-9a-zA-Z\_]{0, 19}
更精确地限制了变量的长度是1-20个字符
（前面1个字符+后面最多19个字符）
'''

# A|B可以匹配A或B，所以(P|p)ython可以匹配'Python'或者'python'
# ^表示行的开头，^\d表示必须以数字开头
# $表示行的结束，\d$表示必须以数字结束
# py也可以匹配'python'，但是加上^py$就变成了整行匹配，就只能匹配'py'了

# 贪婪匹配：
# \d+采用贪婪匹配，直接把后面的0全部匹配了，结果0*只能匹配空字符串了
# >>> re.match(r'^(\d+)(0*)$', '102300').groups()
# ('102300', '')
#
# 必须让\d+采用非贪婪匹配（也就是尽可能少匹配），才能把后面的0匹配出来，加个?就可以让\d+采用非贪婪匹配
# >>> re.match(r'^(\d+?)(0*)$', '102300').groups()
# ('1023', '00')

# # 编译:
# >>> re_telephone = re.compile(r'^(\d{3})-(\d{3,8})$')
# #使用：
# >>> re_telephone.match('010-12345').groups()
# ('010', '12345')

# 练习：请尝试写一个验证Email地址的正则表达式。版本一应该可以验证出类似的Email：
# someone@gmail.com
# bill.gates@microsoft.com
import re


def is_valid_email(addr):
    if(re.match(r'^[a-z]+\.?[a-z]+\@[a-z]+\.com', addr)):
        return True
    else:
        return False


# 测试:


assert is_valid_email('someone@gmail.com')
assert is_valid_email('bill.gates@microsoft.com')
assert not is_valid_email('bob#example.com')
assert not is_valid_email('mr-bob@example.com')
print('ok\n')


# 练习2：
# 版本二可以提取出带名字的Email地址：
# <Tom Paris> tom@voyager.org => Tom Paris
# bob@example.com => bob
def name_of_email(addr):
    ls = re.match(
        r'<?([A-Za-z]+\s?[A-Za-z]*)>?\s?([a-z]*)@([a-z]+)(.org)$', addr)
    if ls:
        return ls.group(1)
    else:
        return None


# 测试:
assert name_of_email('<Tom Paris> tom@voyager.org') == 'Tom Paris'
assert name_of_email('tom@voyager.org') == 'tom'
print('ok')
