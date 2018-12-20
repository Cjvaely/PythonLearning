#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/15 21:21
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : SetSpines.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(1, 5, 50)
y1 = np.power(2.5, x)
y2 = 4 * x
y3 = x ** 2

plt.figure()
plt.plot(x, y1)
# plt.plot(x, y2)
# plt.plot(x, y3)
plt.xlim((-1, 10))
plt.ylim((-2, 50))

# 设置刻度
plt.xticks([1, 3, 5, 7, 9],['$a$', '$b$', '$c$', '$d$', '$e$'])
plt.yticks([10, 20, 30, 40, 50],['$f$', '$g$', '$h$', '$i$', '$j$'])
# x_ticks = np.linspace(-1, 2, 5)
# y_ticks = np.linspace(-1, 3, 8)
# plt.xticks(x_ticks)
# plt.yticks(y_ticks)
# 获取当前坐标轴信息
ax = plt.gca()
# .spines设置边框：右侧边框
# 使用.set_color设置边框颜色：默认白色
# 使用.spines设置边框：上边框
# 使用.set_color设置边框颜色：默认白色
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# .xaxis.set_ticks_position设置x坐标刻度数字或名称的位置：bottom
# 所有位置：top，bottom，both，default，none
ax.xaxis.set_ticks_position('bottom')
# .spines设置边框：x轴
# 使用.set_position设置边框位置：y=0的位置
# 位置所有属性：outward，axes，data
ax.spines['bottom'].set_position(('data', 0))

# .yaxis.set_ticks_position设置y坐标刻度数字或名称的位置：left
# 所有位置：left，right，both，default，none
ax.yaxis.set_ticks_position('left')
# .spines设置边框：y轴
# 使用.set_position设置边框位置：x=0的位置
# 位置所有属性：outward，axes，data, 使用plt.show显示图像
ax.spines['left'].set_position(('data',0))
#"-|>"代表实心箭头："->"代表空心箭头
# ax.axis["bottom"].set_axisline_style("-|>", size = 1.5)
# ax.axis["left"].set_axisline_style("->", size = 1.5)
plt.show()