#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-09 15:39:37
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# psutil:获取系统信息,process and system utilities
import psutil

# 获取CPU的信息：
lognum_cpu = psutil.cpu_count()  # CPU逻辑数量
psynum_cpu = psutil.cpu_count(logical=False)  # CPU物理核心
print('lognum_cpu = %s' % lognum_cpu)
print('psynum_cpu = %s' % psynum_cpu)
# 统计CPU的用户／系统／空闲时间
freetime = psutil.cpu_times()
print('freetime: ', freetime)
# 实现类似top命令的CPU使用率，每秒刷新一次，累计10次：
for x in range(10):
    print(psutil.cpu_percent(interval=1, percpu=True))

# 获取物理内存和交换内存信息，分别使用：
svmem = psutil.virtual_memory()
print('svmem: ', svmem)

sswap = psutil.swap_memory()
print('sswap: ', sswap)
'''.......'''
