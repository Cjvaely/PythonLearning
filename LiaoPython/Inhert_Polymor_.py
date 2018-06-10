#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-31 20:02:44
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# 继承和多态：当我们定义一个class的时候，
# 可以从某个现有的class继承，新的class称为子类（Subclass），
# 而被继承的class称为基类、父类或超类（Base class、Super class）
#
# 编写了一个名为Animal的class，有一个run()方法可以直接打印：

# class Animal(object):
#     def run(self):
#         print('Animal is running...')

# 当我们需要编写Dog和Cat类时，就可以直接从Animal类继承：

# class Dog(Animal):
#     pass

# class Cat(Animal):
#     pass

# 对于Dog来说，Animal就是它的父类，
# 对于Animal来说，Dog就是它的子类。Cat和Dog类似
#
# 继承最大的好处是：子类获得了父类的全部功能
#
# 第二个好处需要我们对代码做一点改进。
# 你看到了，无论是Dog还是Cat，它们run()的时候，
# 显示的都是Animal is running...，
# 符合逻辑的做法是分别显示Dog is running...和Cat is running...，
# 因此，对Dog和Cat类改进如下：

# class Dog(Animal):

#     def run(self):
#         print('Dog is running...')


# class Cat(Animal):

#     def run(self):
#         print('Cat is running...')

# 当子类和父类都存在相同的run()方法时，我们说，子类的run()覆盖了父类的run()，
# 在代码运行的时候，总是会调用子类的run()。这样，我们就获得了继承的另一个好处：多态。

# 当我们定义一个class的时候，我们实际上就定义了一种数据类型。我们定义的数据类型和Python自带的数据类型，比如str、list、dict没什么两样：
# a = list()  # a是list类型
# b = Animal()  # b是Animal类型
# c = Dog()  # c是Dog类型
#
# 在继承关系中，如果一个实例的数据类型是某个子类，那它的数据类型也可以被看做是父类。但是，反过来就不行：
# >>> b = Animal()
# >>> isinstance(b, Dog)
# False
#
# Dog可以看成Animal，但Animal不可以看成Dog。

# 静态语言 vs 动态语言
#
# 对于静态语言（例如Java）来说，如果需要传入Animal类型，
# 则传入的对象必须是Animal类型或者它的子类，否则，将无法调用run()方法。
# 对于Python这样的动态语言来说，则不一定需要传入Animal类型。我们只需要保证传入的对象有一个run()方法就可以了。
