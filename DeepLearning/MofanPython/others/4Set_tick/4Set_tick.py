#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/15 20:51
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : 4Set_tick.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y1 = 2 * x + 1
y2 = x ** 2

plt.figure()
plt.plot(x, y2)
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
# 设置x,y轴坐标名称
plt.xlabel('I am x')
plt.ylabel('I am y')
# 设置x轴刻度范围（-1，2）
new_ticks = np.linspace(-1, 2, 5)
print(new_ticks)
plt.xticks(new_ticks)

# y 的每个刻度对应
plt.yticks([-2, -1.8, -1, 1.22, 3],
           [r'$really\ bad$', r'$bad$',
            r'$normal$', r'$good$', r'$really\ good$'])
plt.show()
