#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-21 21:35:19
# @Author  : Chen Jing (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

height = 1.75
weight = 80.5
bmi = weight / (height * height)
if bmi < 18.5:
	print('过轻！')
elif 18.5 <= bmi <= 25:
	print('正常。')
elif 25 < bmi < 28:
	print('过重！')
elif 28 <= bmi <= 32:
	print('肥胖！')
else:
	print('严重肥胖！')
