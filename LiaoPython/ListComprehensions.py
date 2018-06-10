#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-22 21:08:29
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# List Comprehensions列表生成式
# 举个例子，要生成list [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]可以用list(range(1, 11))
# >>> list(range(1, 11))
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 生成[1x1, 2x2, 3x3, ..., 10x10]怎么做？
# 方法一：循环
# >>> L = []
# >>> for x in range(1, 11):
# ...    L.append(x * x)
# ...
# >>> L
# [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
#
# 方法二：列表生成式
# >>> [x * x for x in range(1, 11)]
# [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
#
# 还可以使用两层循环，可以生成全排列：
#
# >>> [m + n for m in 'ABC' for n in 'XYZ']
# ['AX', 'AY', 'AZ', 'BX', 'BY', 'BZ', 'CX', 'CY', 'CZ']

# 用列表生成式，可以写出非常简洁的代码。例如，列出当前目录下的所有文件和目录名，
# 可以通过一行代码实现：
# >>> import os # 导入os模块，模块的概念后面讲到
# >>> [d for d in os.listdir('.')] # os.listdir可以列出文件和目录
# ['.emacs.d', '.ssh', '.Trash', 'Adlm', 'Applications', 'Desktop', 'Documents', 'Downloads', 'Library', 'Movies', 'Music', 'Pictures', 'Public', 'VirtualBox VMs', 'Workspace', 'XCode']

# for循环其实可以同时使用两个甚至多个变量，
# 比如dict的items()可以同时迭代key和value
# >>> d = {'x': 'A', 'y': 'B', 'z': 'C' }
# >>> for k, v in d.items():
# ...     print(k, '=', v)
# ...
# y = B
# x = A
# z = C

# 因此，列表生成式也可以使用两个变量来生成list：
# >>> d = {'x': 'A', 'y': 'B', 'z': 'C' }
# >>> [k + '=' + v for k, v in d.items()]
# ['y=B', 'x=A', 'z=C']
