# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 15:42:46 2018

@author: cjvae
Learn numpy
"""
import numpy as np
array = np.array([[1, 2, 3], [2, 3, 4]])
print(array)
print('Number of dim:', array.ndim)     # 维度
print('Shape:', array.shape)            # 行数和列数
print('Size:', array.size)              # 元素个数

# 创建数组
a = np.array([2, 23, 4])
print(a)

# 指定数据 dtype
b = np.array([2, 23, 4], dtype = np.int)
print(b.dtype)

c = np.array([2, 23, 4], dtype = np.int64)
print(c.dtype)

# 创建全零数组
d = np.zeros((3, 4))                      # 数据为0， 3行4列
print('Zeros array:\n', d)

# 创建全1数组
e = np.ones((3, 4), dtype = np.int)
print('Ones array:\n', e)

# 创建全空数组
f = np.empty((3, 4))
print('Empty array:\n', f)

# 用 arange 创建连续数组， 10-19， 步长2为2
g = np.arange(10, 20, 2)
print('Go-on array:\n', g)

# 使用 reshape 改变数据的形状
h = np.arange(12).reshape((4, 3))
print('Reshape array:\n', h)

# 用 linspace 创建线段型数据
# 开始端1，结束端10，且分割成20个数据，生成线段
i = np.linspace(1, 10, 20).reshape((4, 5))
print('Linspace array:\n', i)


# 基础运算
a1 = np.array([10, 20, 30, 40])
b1 = np.arange(4)
# 矩阵加减法
c1 = a1 - b1 
print('After sub:\n', c1)

# 矩阵乘法
d1 = a1 * b1
print('After mul:\n', d1)

# 矩阵乘方
e1 = b1 ** 3
print('After multia mul:\n', e1)

# 三角函数
f1 = 10 * np.sin(a1)
print('After sin function:\n', f1)

# 对print函数进行一些修改可以进行逻辑判断
print(b1 < 3)

# 多维矩阵操作
h1 = np.array([[1, 1], [0, 1]])
i1 = np.arange(4).reshape((2, 2))
# 对应行乘对应列得到相应元素
c_dot = np.dot(h1, i1)
print('Second mul:\n', c_dot)

# sum() min() max()
a2 = np.random.random((2, 4))
print('Random narray:\n', a2)
print('The sum of random array:\n', np.sum(a2))
print('The max number of array:\n', np.max(a2))
print('The min number of array:\n', np.min(a2))

# 当axis的值为0的时候，将会以列作为查找单元
# 当axis的值为1的时候，将会以行作为查找单元
print("sum =",np.sum(a2, axis=1))
print("min =",np.min(a2,axis=0))
print("max =",np.max(a2,axis=1))

# 矩阵的索引运算
A = np.arange(2, 14).reshape((3, 4))
print('The index of min number:\n', np.argmin(A))
print('The index of max number:\n', np.argmax(A))

# 求均值
print('The mean of array:\n', np.mean(A))
print('The average of array:\n', np.average(A))

# 求中位数
print('Middle num:\n', np.median(A))

# cumsum 累加运算
print('Cumsum:\n', np.cumsum(A).reshape((3, 4)))
# diff 累差运算
print('累差:\n', np.diff(A))

# nonzero()函数:得到数组array中非零元素的位置
print("The using of nonzero:\n", np.nonzero(A))

# 排序
B = np.arange(14, 2, -1).reshape((3, 4))
print('Before sort:\n', B)
print('After sort:\n', np.sort(A))

# 矩阵转置
print('The first transpose:\n', np.transpose(B))
print('The second transpose:\n', B.T)

# clip(Array,Array_min,Array_max)
print('After clip:\n', np.clip(B, 5, 9))

# numpy一维索引
C = np.arange(3, 15)
print('Using one dim array index:\n', C[3])
C = np.arange(3, 15).reshape((3, 4))
print('Using two dim array index:\n', C[2])

# numpy 二维索引
print('Using two dim  index:\n', C[2][2])

# 切片操作
print('Slice in numpy:\n', C[1, 1:3])

# 逐行打印
print('逐行打印:')
for row in C:
    print(row)
    
# 逐列打印
print('逐列打印:')
for column in C.T:
    print(column)

# 迭代输出, flat是迭代器
print('C = ', C)
print('Array flatten:', C.flatten())
print("迭代器输出：")
for item in C.flat:
    print(item)

# Numpy array 合并
A2 = np.array([1, 1, 1, 1])
B2 = np.array([2, 2, 2, 2])
# 上下合并
print('上下合并两个数组：\n', np.vstack((A2, B2)))
C2 = np.vstack((A2, B2))
print('A2 shape = ', A2.shape)
print('C2 shape = ', C2.shape)
#左右合并
D2 = np.hstack((A2, B2))
print("左右合并两个数组：\n ", D2)
print('D2 shape:', D2.shape)

# 序列转置
print('序列转置：\n')
print('A2 = ', A2)
print('A2[np.newaxis, :]\n', A2[np.newaxis, :])
print('A2[:, np.newaxis]\n', A2[:, np.newaxis])

# mpy array 分割
A3 = np.arange(12).reshape((3, 4))
print('A3 = ', A3)
# 纵向分割
print('纵向分割：\n', np.split(A3, 2, axis = 1))
# 横向分割 
print('横向分割：\n', np.split(A3, 3, axis = 0))
#不等量的分割
print('不等量分割：\n', np.array_split(A3, 3, axis = 1))
# 其它的分割方式
print('横向切割2.0：\n', np.vsplit(A3, 3))
print('纵向切割2.0：\n', np.hsplit(A3, 2))


# Numpy copy & deep copy

# = 的赋值方式会带有关联性
B3 = np.arange(4)
C3 = B3
D3 = B3
E3 = C3
B3[0] = 11
print('B3 = ', B3)
print('C3 = ', C3)
print('D3 = ', D3)
D3[1:3] = [22, 33]
print('B3 = ', B3)
print('C3 = ', C3)
print('D3 = ', D3)

#copy() 的赋值方式没有关联性
C3 = B3.copy()
print('C3 = ', C3)
B3[3] = 44
print('New B3 = ', B3)
print('New C3 = ', C3)









