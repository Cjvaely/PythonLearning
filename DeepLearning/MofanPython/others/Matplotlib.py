# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:12:12 2018

@author: cjvae
Learn matplotlib
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib import animation

# # 定义x：范围是(-1,1);个数是50
# x = np.linspace(-1, 1, 50)
# y = 2 * x + 1
# # plt.figure()
# plt.plot(x, y)
# # plt.show()

# # figure 图像

# # 简单线条，figure就是一个单独的figure小窗口
# x = np.linspace(-3, 3, 50)
# y1 = 2 * x + 1
# y2 = x ** 2
# # 定义一个图像窗口
# # plt.figure()
# plt.plot(x, y1)
# # plt.show()

# # 定义窗口编号和大小
# # plt.figure(num=3, figsize=(8, 5))
# plt.plot(x, y2)
# plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
# # plt.show()

# # 设置坐标轴
# # 调整名字和间隔
# x = np.linspace(-3, 3, 50)
# y1 = 2 * x + 1
# y2 = x ** 2

# # plt.figure()
# plt.plot(x, y2)
# plt.plot(x, y1, color='yellow', linewidth=1.0, linestyle='--')
# # x轴范围 （-1,2）, y轴范围（-2, 3）
# plt.xlim(-1, 2)
# plt.ylim(-2, 3)
# # s设置坐标轴名称
# plt.xlabel('I am x')
# plt.ylabel('I am y')
# # plt.show()

# # np.linspace定义范围以及个数，plt.xticks设置x轴刻度
# # plt.yticks设置y轴刻度以及名称
# new_ticks = np.linspace(-1, 2, 5)
# # print(new_ticks)
# plt.xticks(new_ticks)
# plt.yticks([-2, -1.8, -1, 1.22, 3],
#            [r'$really\ bad$', r'$bad$',
#             r'$normal$', r'$good$', r'$really\ good$'])
# # plt.show()

# # 设置不同名字和位置
# plt.figure()
# plt.plot(x, y2)
# # 颜色属性(color)为红色,曲线的宽度(linewidth)为1.0,
# # 曲线的类型(linestyle)为虚线
# plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
# # 设置坐标轴范围
# plt.xlim((-1, 2))
# plt.ylim((-2, 3))
# # np.linspace定义范围以及个数：范围是(-1,2);个数是5
# new_ticks = np.linspace(-1, 2, 5)
# plt.xticks(new_ticks)
# plt.yticks([-2, -1.8, -1, 1.22, 3],
#            [r'$really\ bad$', r'$bad$',
#             r'$normal$', r'$good$', r'$really\ good$'])
# # plt.gca获取当前坐标轴信息, .spines设置边框, .set_color设置边框颜色
# ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# plt.show()

# # 调整坐标轴
# # .xaxis.set_ticks_position设置x坐标刻度数字或名称的位置：bottom
# # （所有位置：top，bottom，both，default，none）
# ax.xaxis.set_ticks_position('bottom')
# # .spines设置边框：x轴；使用.set_position设置边框位置：y=0的位置
# # 位置所有属性：outward，axes，data
# ax.spines['bottom'].set_position(('data', 0))
# plt.show()


# # 添加图例
# x = np.linspace(-3, 3, 50)
# y1 = 2 * x + 1
# y2 = x**2

# plt.figure()
# # set x limits
# plt.xlim((-1, 2))
# plt.ylim((-2, 3))

# # set new sticks
# new_sticks = np.linspace(-1, 2, 5)
# plt.xticks(new_sticks)
# # set tick labels
# plt.yticks([-2, -1.8, -1, 1.22, 3],
#            [r'$really\ bad$', r'$bad$',
#             r'$normal$', r'$good$', r'$really\ good$'])
# # set line style
# l1 = plt.plot(x, y1, label='linear line')
# l2 = plt.plot(x, y2, color='red', linewidth=1.0,
#               linestyle='--', label='square line')
# # 添加在图中的右上角.
# plt.legend(loc='best')
# # plt.show()


# Annotation 标注

# 坐标轴中绘制一条直线
# a = np.linspace(-3, 3, 50)
# b = 2 * a + 1
# plt.figure(num=1, figsize=(8, 5),)
# plt.plot(a, b)
# # plt.show()

# # 移动坐标
# ax = plt.gca()
# # 若不设置，会导致存在右边框
# ax.spines['right'].set_color('none')
# # 若不设置，会导致存在上边框
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# # 把y = 0放到底端
# ax.spines['bottom'].set_position(('data', 0))
# ax.yaxis.set_ticks_position('left')
# # 把x = 0放到0的位置
# ax.spines['left'].set_position(('data', 0))
# # plt.show()

# # 画出一条垂直于x轴的虚线
# x0 = 1
# y0 = 2 * x0 + 1
# plt.plot([x0, x0, ], [0, y0], 'k--', linewidth=2.5)
# # set dot styles
# plt.scatter([x0, ], [y0, ], s=50, color='b')
# plt.show()

# 添加注释 annotate
# 对(x0, y0进行标注)
# xycoords='data' 是说基于数据的值来选位置
# xytext=(+30, -30) 和 textcoords = 'offset points'
# 标注位置的描述 和 xy 偏差值
# arrowprops是对图中箭头类型的一些设置.

# plt.annotate(r'$2x+1=%s$' % y0, xy=(x0, y0),
#              xycoords='data', xytext=(+30, -30),
#              textcoords='offset points', fontsize=16,
#              arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
# plt.show()

# 添加注释 text
# -3.7, 3,是选取text的位置
# 空格需要用到转字符\ ,fontdict设置文本字体
# plt.text(-3.7, 3, r'$This\ is\ the\ some\ text. \mu\ \sigma_i\ \alpha_t$',
#          fontdict={'size': 16, 'color': 'r'})
# plt.show()


# tick 能见度
# x = np.linspace(-3, 3, 50)
# y = 0.1 * x

# plt.figure()
# # 在 plt 2.0.2 或更高的版本中, 设置 zorder 给 plot 在 z 轴方向排序
# plt.plot(x, y, linewidth=10, zorder=1)
# plt.ylim(-2, 2)
# ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.spines['bottom'].set_position(('data', 0))
# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data', 0))

# 调整坐标 对x轴和y轴的刻度做透明度设置
# label.set_fontsize(12)重新调节字体大小
# bbox设置目的内容的透明度
# facecolor调节 box 前景色
# edgecolor 设置边框
# alpha设置透明度
# for label in ax.get_xticklabels() + ax.get_yticklabels():
#     label.set_fontsize(12)
#     # 在 plt 2.0.2 或更高的版本中, 设置 zorder 给 plot 在 z 轴方向排序
#     label.set_bbox(
#         dict(facecolor='white', edgecolor='None', alpha=0.7, zorder=2))
# plt.show()


# Scatter 散点图
"""numpy用来产生一些随机数据
1024个呈标准正态分布的二维数据组 (平均数是0，方差为1) 作为一个数据集
图像化这个数据集
每一个点的颜色值用T来表示
"""
# n = 1024
# X = np.random.normal(0, 1, n)  # 每一个点的X值
# Y = np.random.normal(1, 1, n)  # 每一个点的X值
# T = np.arctan2(Y, X)  # for color value
# # X和Y作为location
# # size=75
# # 颜色为T
# # color map用默认值
# # 透明度alpha 为 50%
# # x轴显示范围定位(-1.5，1.5)
# # xtick()函数来隐藏x坐标轴，y轴
# plt.scatter(X, Y, s=75, c=T, alpha=.5)

# plt.xlim(-1.5, 1.5)
# plt.xticks(())
# plt.ylim(-1.5, 1.5)
# plt.yticks(())
# plt.show()


# Bar 柱状图
# n = 12
# # X为 0 到 11 的整数
# # Y是相应的均匀分布的随机数据
# X = np.arange(n)
# Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
# Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)

# plt.bar(X, +Y1)
# plt.bar(X, -Y2)

# plt.xlim(-.5, n)
# plt.xticks(())
# plt.ylim(-1.25, 1.25)
# plt.yticks(())
# plt.show()
# 加颜色和数据

# facecolor设置主体颜色
# edgecolor设置边框颜色为白色
# plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
# plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')
# # plt.show()

# # plt.text分别在柱体上方（下方）加上数值, %.2f保留两位小数
# # 横向居中对齐ha='center'
# # 纵向底部（顶部）对齐va='bottom'
# for x, y in zip(X, Y1):
#     # ha: horizontal alignment
#     # va: vertical alignment
#     plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')
# for x, y in zip(X, Y2):
#     plt.text(x + 0.4, -y - 0.05, '%.2f' % y, ha='center', va='top')
# plt.show()


# 等高线图
# 画等高线

# 函数生成高度值
# def f(x, y):
#     return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)


# n = 256
# x = np.linspace(-3, 3, n)
# y = np.linspace(-3, 3, n)
# # meshgrid在二维平面中将每一个x和每一个y分别对应起来，编织成栅格:
# X, Y = np.meshgrid(x, y)
# # use plt.contourf to filling contours
# # X, Y and value for (X,Y) point
# #  f(X,Y) 的值对应到color map的暖色组中寻找对应颜色
# plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap=plt.cm.hot)
# # use plt.contour to add contour lines
# C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5)
# # 添加高度数字
# plt.clabel(C, inline=True, fontsize=10)
# plt.xticks(())
# plt.yticks(())
# plt.show()


# Image 图片
# 随机矩阵画图
# a = np.array([0.313660827978, 0.365348418405, 0.423733120134,
#               0.365348418405, 0.439599930621, 0.525083754405,
#               0.423733120134, 0.525083754405, 0.651536351379]).reshape(3, 3)

# # 三行三列的格子，a代表每一个值，图像右边有一个注释，
# # 白色代表值最大的地方，颜色越深值越小。

# # origin='lower'代表的就是选择的原点的位置。
# plt.imshow(a, interpolation='nearest', cmap='bone', origin='lower')
# # colorbar,shrink参数，使colorbar的长度变短为原来的92%
# plt.colorbar(shrink=.92)
# plt.xticks(())
# plt.yticks(())
# plt.show()


# 3D 数据, 额外添加一个模块
# from mpl_toolkits.mplot3d import Axes3D
# 先定义一个图像窗口，在窗口上添加3D坐标轴
# fig = plt.figure()
# ax = Axes3D(fig)
# # x y values
# X = np.arange(-4, 4, 0.25)
# Y = np.arange(-4, 4, 0.25)
# X, Y = np.meshgrid(X, Y)  # x-y 平面的网格
# # Z 每一个（X, Y）点对应的高度
# R = np.sqrt(X ** 2 + Y ** 2)
# Z = np.sin(R)
# # colormap rainbow 填充颜色
# # rstride 和 cstride 分别代表 row 和 column 的跨度
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
# # plt.show()

# # 投影
# # 添加 XY 平面的等高线
# # zdir 选择了x，那么效果将会是对于 XZ 平面的投影
# ax.contour(X, Y, Z, zdir='z', offset=2, cmap=plt.get_cmap('rainbow'))
# plt.show()


# Subplot 多合一显示
# 均匀图中图
# 组合许多的小图, 放在一张大图里面显示的. 使用到的方法叫作 subplot
# plt.figure()
# # plt.subplot来创建小图
# # plt.subplot(2,2,1)表示将整个图像窗口分为2行2列,当前位置为1
# plt.subplot(2, 2, 1)
# # plt.plot([0, 1], [0, 1])在第1个位置创建一个小图
# plt.plot([0, 1], [0, 1])
# # plt.subplot(2,2,2)表示将整个图像窗口分为2行2列, 当前位置为2
# plt.subplot(2, 2, 2)
# # plt.plot([0,1],[0,2])在第2个位置创建一个小图.
# plt.plot([0, 1], [0, 2])
# plt.subplot(223)
# plt.plot([0, 1], [0, 3])
# plt.subplot(224)
# plt.plot([0, 1], [0, 4])
# plt.show()

# # 把第1个小图放到第一行, 而剩下的3个小图都放到第二行
# # 分为2行1列, 当前位置为1
# # 第1个位置创建一个小图
# plt.subplot(2, 1, 1)
# plt.plot([0, 1], [0, 1])
# # 2行3列, 当前位置为4
# plt.subplot(2, 3, 4)
# # 第4个位置创建一个小图
# plt.plot([0, 1], [0, 2])
# """
#     上一步中使用plt.subplot(2,1,1)将整个图像窗口分为2行1列
#     第1个小图占用了第1个位置, 也就是整个第1行
#      这一步中使用plt.subplot(2,3,4)将整个图像窗口分为2行3列,
#       于是整个图像窗口的第1行就变成了3列
#       于是第2行的第1个位置是整个图像窗口的第4个位置

# """
# # 2行3列,当前位置为5.
# plt.subplot(235)
# # 第5个位置创建一个小图
# plt.plot([0, 1], [0, 3])
# plt.subplot(236)
# plt.plot([0, 1], [0, 4])
# plt.show()


# Subplot 分格显示
# 1、 subplot2grid
# plt.figure()
# # plt.subplot2grid来创建第1个小图
# # (3,3)表示将整个图像窗口分成3行3列
# # (0,0)表示从第0行第0列开始作图
# # colspan=3表示列的跨度为3, rowspan=1表示行的跨度为1
# ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
# ax2 = plt.plot([1, 2], [1, 2])
# ax1.set_title('ax1_title')  # 设置标题

# ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
# ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
# ax4 = plt.subplot2grid((3, 3), (2, 0))
# ax5 = plt.subplot2grid((3, 3), (2, 1))
# # ax4.scatter创建一个散点图
# ax4.scatter([1, 2], [2, 2])
# # 对x轴和y轴命名
# ax4.set_xlabel('ax4_x')
# ax4.set_ylabel('ax4_y')
# plt.show()

# gridspec
# import matplotlib.gridspec as gridspec
# plt.figure()
# gs = gridspec.GridSpec(3, 3)
# """gs[0, :]表示这个图占第0行和所有列
# gs[1, :2]表示这个图占第1行和第2列前的所有列
# gs[1:, 2]表示这个图占第1行后的所有行和第2列
# gs[-1, -2]表示这个图占倒数第1行和倒数第2列
# """
# ax6 = plt.subplot(gs[0, :])
# ax7 = plt.subplot(gs[1, :2])
# ax8 = plt.subplot(gs[1:, 2])
# ax9 = plt.subplot(gs[-1, 0])
# ax10 = plt.subplot(gs[-1, -2])
# plt.show()


# Animation 动画
fig, ax = plt.subplots()
# 一个0~2π内的正弦曲线
x = np.arange(0, 2 * np.pi, 0.01)
line, = ax.plot(x, np.sin(x))


# 自定义动画函数animate, 更新每一帧上各个x对应的y坐标值
# 参数表示第i帧
def animate(i):
    line.set_ydata(np.sin(x + i / 10.0))
    return line,


# 构造开始帧函数init
def init():
    line.set_ydata(np.sin(x))
    return line,


"""fig 进行动画绘制的figure
func 自定义动画函数，即传入刚定义的函数animate
frames 动画长度，一次循环包含的帧数
init_func 自定义开始帧，即传入刚定义的函数init
interval 更新频率，以ms计
blit 选择更新所有点，还是仅更新产生变化的点。
应选择True，但mac用户请选择False，否则无法显示动画
"""
ani = animation.FuncAnimation(fig=fig,
                              func=animate,
                              frames=100,
                              init_func=init,
                              interval=20,
                              blit=False)
plt.show()
