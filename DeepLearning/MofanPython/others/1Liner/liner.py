#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-15 19:59:40
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : http://www.cnblogs.com/cjvae/
# @Version : $Id$

import matplotlib.pyplot as plt
import numpy as np

# 定义x：范围是(-1,1);个数是50
x = np.linspace(-1, 1, 50)
y = 2 * x + 1
plt.figure()
plt.plot(x, y)
plt.show()
