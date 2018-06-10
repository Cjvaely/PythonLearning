#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-30 20:34:30
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# 偏函数 Partial function:
# 通过设定参数的默认值，可以降低函数调用的难度。而偏函数也可以做到这一点。举例如下：
# int()函数可以把字符串转换为整数，当仅传入字符串时，int()函数默认按十进制转换：
# >> > int('12345')
# 12345
#
# 但int()函数还提供额外的base参数，默认值为10。如果传入base参数，就可以做N进制的转换：
# >>> int('12345', base=8)
# 5349
# >>> int('12345', 16)
# 74565

# functools.partial就是帮助我们创建一个偏函数的，不需要我们自己定义int2()，可以直接使用下面的代码创建一个新的函数int2：
# >>> import functools
# >>> int2 = functools.partial(int, base=2)
# >>> int2('1000000')
# 64
# >>> int2('1010101')
# 85

# 创建偏函数时，实际上可以接收函数对象、*args和**kw这3个参数
# 当函数的参数个数太多，需要简化时，使用functools.partial可以创建一个新的函数，
# 这个新函数可以固定住原函数的部分参数，从而在调用时更简单。
#
