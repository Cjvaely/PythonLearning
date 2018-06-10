#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-02 15:06:03
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# Practice Enum
# 把Student的gender属性改造为枚举类型，可以避免使用字符串：
from enum import Enum, unique


@unique
class Gender(Enum):
    Male = 0
    Female = 1


class Student(object):
    gender = Gender.Male

    def __init__(self, name, gender):
        self.name = name
        self.gender = gender


# 测试:
bart = Student('Bart', Gender.Male)
if bart.gender == Gender.Male:
    print('测试通过!')
else:
    print('测试失败!')
