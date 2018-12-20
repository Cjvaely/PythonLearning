# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 23:01:06 2018

@author: cjvae
Learn Pandas
"""
import pandas as pd
import numpy as np

# 两个数据结构：Series和DataFrame
# Seeries, 索引在左边，值在右边
s = pd.Series([1, 3, 6, np.nan, 44, 1])
print('s = \n', s)

# DataFrame, 表格型的数据结构，它包含有一组有序的列，
# 每列可以是不同的值类型（数值，字符串，布尔值等）
dates = pd.date_range('20160101',periods=6)
df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=['a','b','c','d'])
print('df = \n',df)

# dataFrame一些简单运用
print('df[b] = \n', df['b'])
# 默认索引， 从0开始
df1 = pd.DataFrame(np.arange(12).reshape((3, 4)))
print('df1 = \n', df1)
# 第二种：对每一列的数据进行特殊对待
df2 = pd.DataFrame({'A':1., 'B':pd.Timestamp('20130102'),
                    'C':pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D':np.array([3] * 4, dtype='int32'),
                    'E':pd.Categorical(['test', 'train', 'test', 'train']),
                    'F':'foo'})
print('df2 = \n', df2)
print('对列序号：\n', df2.index)
print('数据的名称:\n', df2.columns)
print('只看值：\n', df2.values)
print('数据的总结：\n', df2.describe())
print('翻转数据：\n', df2.T)

# Pandas 选择数据
# 建立一个6x4矩阵
Datas = pd.date_range('20130101', periods=6)
Df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates,columns=['A','B','C','D'])
print('Df = \n', Df)
print("Df['A'] = \n", Df['A'])
print('Df.A = \n', Df.A)





