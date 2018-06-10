#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-22 21:25:01
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# 列表生成式练习

# 请修改列表生成式，通过添加if语句保证列表生成式能正确地执行：

L1 = ['Hello', 'World', 18, 'Apple', None]
L2 = []
for x in L1:
    if isinstance(x, str):
        L2.append(x)
L2 = [s.lower() for s in L2]
# 测试:
print(L2)
if L2 == ['hello', 'world', 'apple']:
    print('测试通过!')
else:
    print('测试失败!')
