#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/15 22:30
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : test.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
import numpy as np

#创建画布
fig = plt.figure()
#使用axisartist.Subplot方法创建一个绘图区对象ax
ax = axisartist.Subplot(fig, 111)
#将绘图区对象添加到画布中
fig.add_axes(ax)
#通过set_axisline_style方法设置绘图区的底部及左侧坐标轴样式
#"-|>"代表实心箭头："->"代表空心箭头
ax.axis["bottom"].set_axisline_style("-|>", size = 1.5)
ax.axis["left"].set_axisline_style("->", size = 1.5)
#通过set_visible方法设置绘图区的顶部及右侧坐标轴隐藏
ax.axis["top"].set_visible(False)
ax.axis["right"].set_visible(False)

plt.xlim((-1, 10))
plt.ylim((-2, 50))

x = np.linspace(1, 5, 50)
y1 = np.power(2.5, x)
# y2 = 4 * x
# y3 = x ** 2
plt.xticks([1, 3, 5, 7, 9],['$a$', '$b$', '$c$', '$d$', '$e$'])
plt.yticks([10, 20, 30, 40, 50],['$f$', '$g$', '$h$', '$i$', '$j$'])
plt.plot(x, y1)
plt.show()