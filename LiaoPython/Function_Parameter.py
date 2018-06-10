#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-22 13:07:04
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# 函数的参数
# 默认参数降低了函数调用的难度，而一旦需要更复杂的调用时，
# 又可以传递更多的参数来实现。无论是简单调用还是复杂调用，
# 函数只需要定义一个
# 定义默认参数要牢记一点：默认参数必须指向不变对象！
# 我们在编写程序时，如果可以设计一个不变对象，那就尽量设计成不变对象。
#
# 可变参数：传入的参数个数是可变的，可以是1个、2个到任意个，还可以是0个
# 由于参数个数不确定，我们首先想到可以把a，b，c……作为一个list或tuple传进来
# def calc(numbers):
#     sum = 0
#     for n in numbers:
#         sum = sum + n * n
#     return sum
# 调用的时候，需要先组装出一个list或tuple:
# >>> calc([1, 2, 3])
# 14
# >>> calc((1, 3, 5, 7))
# 84

# def calc(*numbers):
#     sum = 0
#     for n in numbers:
#         sum = sum + n * n
#     return sum
# 如果利用可变参数，调用函数的方式可以简化成这样：
# >>> calc(1, 2, 3)
# 14
# >>> calc(1, 3, 5, 7)
# 84
# 定义可变参数和定义一个list或tuple参数相比，仅仅在参数前面加了一个*号。
# 在函数内部，参数numbers接收到的是一个tuple，因此，函数代码完全不变
#
# 如果已经有一个list或者tuple，要调用一个可变参数怎么办？
# >>> nums = [1, 2, 3]
# >>> calc(nums[0], nums[1], nums[2])
# 14
# 但是太繁琐，可以换成：
# 在list或tuple前面加一个*号，把list或tuple的元素变成可变参数传进去
#
# 关键字参数：允许你传入0个或任意个含参数名的参数，这些关键字参数在函数内部自动组装为一个dict。
# 请看示例：
# def person(name, age, **kw):
#     print('name:', name, 'age:', age, 'other:', kw)
# 函数person除了必选参数name和age外，还接受关键字参数kw。在调用该函数时，可以只传入必选参数
# >>> person('Michael', 30)
# name: Michael age: 30 other: {}
#
# 也可以传入任意个数的关键字参数：
# >>> person('Adam', 45, gender='M', job='Engineer')
# name: Adam age: 45 other: {'gender': 'M', 'job': 'Engineer'}
#
# 和可变参数类似，也可以先组装出一个dict，然后，把该dict转换为关键字参数传进去：
# >>> extra = {'city': 'Beijing', 'job': 'Engineer'}
# >>> person('Jack', 24, city=extra['city'], job=extra['job'])
# name: Jack age: 24 other: {'city': 'Beijing', 'job': 'Engineer'}
#
# 命名关键字参数：如果要限制关键字参数的名字，就可以用命名关键字参数，
# 例如，只接收city和job作为关键字参数。这种方式定义的函数如下：
# def person(name, age, *, city, job):
# 	print(name, age, city, job)
# 	和关键字参数**kw不同，命名关键字参数需要一个特殊分隔符*，
# 	*后面的参数被视为命名关键字参数。
#
# 	如果函数定义中已经有了一个可变参数，后面跟着的命名关键字参数就不再需要一个特殊分隔符*了：
# def person(name, age, *args, city, job):
#     print(name, age, args, city, job)
#
# 参数组合：这5种参数都可以组合使用。但是请注意，参数定义的顺序必须是：必选参数、默认参数、可变参数、命名关键字参数和关键字参数。
# 例子：
# def f1(a, b, c=0, *args, **kw):
#     print('a =', a, 'b =', b, 'c =', c, 'args =', args, 'kw =', kw)

# def f2(a, b, c=0, *, d, **kw):
# 	print('a =', a, 'b =', b, 'c =', c, 'd =', d, 'kw =', kw)
