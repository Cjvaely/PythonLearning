#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-02 14:08:12
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# Using @property
# Python内置的@property装饰器就是负责把一个方法变成属性调用的：
#
# 练习：利用@property给一个Screen对象加上width和height属性，
# 以及一个只读属性resolution


class Screen(object):
    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, w):
        self._width = w

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, h):
        self._height = h

    @property
    def resolution(self):
        self._resolution = self._width * self._height
        return self._resolution


# 测试:
s = Screen()
s.width = 1024
s.height = 768
print('resolution =', s.resolution)
if s.resolution == 786432:
    print('测试通过!')
else:
    print('测试失败!')
