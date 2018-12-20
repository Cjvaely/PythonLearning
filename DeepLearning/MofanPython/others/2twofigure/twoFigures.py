#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-15 20:00:19
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : http://www.cnblogs.com/cjvae/
# @Version : $Id$

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y1 = 2 * x + 1
y2 = x ** 2

# 定义窗口编号为3和大小（8，5）
plt.figure(num = 3, figsize = (8, 5))
# plot画线x,y2
plt.plot(x, y2)
# plot画线x, y1。color属性，曲线的宽度,曲线的类型
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
plt.show()
