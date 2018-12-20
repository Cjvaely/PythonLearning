#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-15 20:20:05
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : http://www.cnblogs.com/cjvae/
# @Version : $Id$

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y1 = 2 * x + 1
y2 = x ** 2

plt.figure()
plt.plot(x, y2)
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')

# 设置x轴，y轴范围
plt.xlim((-1, 2))
plt.ylim((-2, 3))
# 设置x,y轴坐标名称
plt.xlabel('I am x')
plt.ylabel('I am y')
plt.show()
