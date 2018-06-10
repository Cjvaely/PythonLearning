#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-24 12:51:30
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# 练习： 回数是指从左向右读和从右向左读都是一样的数，例如12321，909。
# 请利用filter()筛选出回数：


def exchange(num):
    count = len(str(num)) - 1  # 数字位数3214 -> 4123
    sum = 0
    m = 0
    while num != 0:
        sum = sum + (num // (10 ** count)) * \
            (10 ** m)  # sum = 3   23  123   4123
        num = num % (10 ** count)  # num = 214  14  4
        count = count - 1  # 2  1   0
        m = m + 1  # 1  2   3
    return sum


def is_palindrome(n):
    count = len(str(n))  # 数字位数
    if count == 1:
        return True
    elif count % 2 == 0:  # 位数为偶数
        half1 = n // 10 ** (count // 2)  # 数字前半部分
        half2 = n % 10 ** (count // 2)  # 数字后半部分
        if half1 == exchange(half2):
            return True
    else:  # 位数非1 且为奇数
        half1 = n // 10 ** ((count + 1) // 2)  # 数字前半部分
        half2 = n % 10 ** ((count - 1) // 2)  # 数字后半部分
        if half1 == exchange(half2):
            return True


# 借鉴切片： return str(n) == str(n)[::-1]
# 测试:
output = filter(is_palindrome, range(1, 1000))
print('1~1000:', list(output))
if list(filter(is_palindrome, range(1, 200))) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 22, 33, 44, 55, 66, 77, 88, 99, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191]:
    print('测试成功!')
else:
    print('测试失败!')
