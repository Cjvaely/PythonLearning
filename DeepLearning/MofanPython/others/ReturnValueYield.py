#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-24 21:17:46
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : http://www.cnblogs.com/cjvae/
# @Version : $Id$

# 关于yield的返回值，send方法：send方法有一个参数，该参数指定的是上一次被挂起的yield语句的返回值。


def func1():  # 生成器函数
    x = yield 1  # 这里执行
    print('This is x in func1: ', x)
    x = yield x
    # print('This is x:', x)


f1 = func1()
# 当调用next(f1)方法时，python首先会执行func1方法的yield 1语句
# 由于是一个yield语句, next方法返回值为yield关键字后面表达式的值，即为1
print('This is next(f1): ', next(f1))
# print('This is second using next:', next(f1))
# 当调用f1.send('e')方法时,python首先恢复func1方法的运行环境.
# 同时，表达式(yield 1)的返回值,即为x, 定义为send方法参数的值（'e'）
print("This is f1.send('e'):", f1.send('e'))
# print("Second next:", next(f1))
# 这样，接下来x =（yield 1）这一赋值语句会将x的值置为'e'。
# 但是，如果此时不调用send方法，之前中断之后的语句不会执行了
# 如果再调用next方法，func1会从之前的中断，继续运行print('This is x in func1: ', x)将被执行。
# 显然，x的值是None。next(f1)返回的也是None。

# 继续运行会遇到语句。
# 这时，fun1方法再次被挂起，同时，send方法的返回值为yield关键字后面表达式的值，也即x的值('e')
print("This is f1.send('f'):", f1.send('f'))
# 当调用send('f')方法时，将表达式(yield x)的返回值定义为send方法参数的值，即为'f'。
# 这样，接下来x = yield x 这一赋值语句会将x的值置为'f'。继续运行，func1方法执行完毕，
# 故而抛出StopIteration异常。

# 总而言之，send方法和next方法的区别在于，执行send方法时，
# 会首先把执行上一次挂起的yield语句的返回值, 通过send方法内的参数设定。
# 但是需要注意，在一个生成器对象没有执行next方法之前，由于没有yield语句被挂起，所以执行send方法会
# f2 = func1()
# print("f2.send('g'):", f2.send('g'))
"""
Traceback (most recent call last):
  File "ReturnValueYield.py", line 35, in <module>
    print("f2.send('g'):", f2.send('g'))
TypeError: can't send non-None value to a just-started generator
"""

# 下面的是可以的：
# f3 = func1()
# print("Send None value: ", f3.send(None))
# Send None value:  1
"""
当send方法的参数为None时，它与next方法完全等价。但是注意，虽然上面的代码可以接受，但是不规范。
所以，在调用send方法之前，还是先调用一次next方法为好。
"""
