#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-22 01:36:41
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# dict dict全称dictionary，在其他语言中也称为map，使用键-值（key-value）存储，具有极快的查找速度
# 给一个例子：
# names = ['Michael', 'Bob', 'Tracy']
# scores = [95, 75, 85]
# 给定一个名字，要查找对应的成绩，就先要在names中找到对应的位置，再从scores取出对应的成绩，list越长，耗时越长。

# 如果用dict实现，只需要一个“名字”-“成绩”的对照表，直接根据名字查找成绩，无论这个表有多大，查找速度都不会变慢。用Python写一个dict如下：

# >>> d = {'Michael': 95, 'Bob': 75, 'Tracy': 85}
# >>> d['Michael']
# 95
# 由于一个key只能对应一个value，所以，多次对一个key放入value，后面的值会把前面的值冲掉
# 如果key不存在，dict就会报错;要避免key不存在的错误，有两种办法，一是通过in判断key是否存在
# >>> 'Thomas' in d
# False
# 二是通过dict提供的get()方法，如果key不存在，可以返回None，或者自己指定的value
# >>> d.get('Thomas')
# 要删除一个key，用pop(key)方法，对应的value也会从dict中删除：
# >>> d.pop('Bob')
# 75
# 
# 和list比较，dict有以下几个特点：
# 查找和插入的速度极快，不会随着key的增加而变慢；
# 需要占用大量的内存，内存浪费多。
# 而list相反：

# 查找和插入的时间随着元素的增加而增加；
# 占用空间小，浪费内存很少。
# 所以，dict是用空间来换取时间的一种方法。
# 正确使用dict非常重要，需要牢记的第一条就是dict的key必须是不可变对象。
# 