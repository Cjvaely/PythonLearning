#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-05 17:23:53
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# IO Programming
# 解决IO速度不匹配问题：

# 第一种是CPU等着，也就是程序暂停执行后续代码，
# 等100M的数据在10秒后写入磁盘，再接着往下执行，
# 这种模式称为同步IO

# 另一种方法是CPU不等待，只是告诉磁盘，“您老慢慢写，不着急，
# 我接着干别的事去了”，于是，后续代码可以立刻接着执行，
# 这种模式称为异步IO

# 使用异步IO来编写程序性能会远远高于同步IO，但是异步IO的缺点是编程模型复杂
