#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018-11-28 09:33
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : Cap_picture.py
# @Software: PyCharm

import cv2

# 设置视频捕获
cap = cv2.VideoCapture(0)
while True:
    ret, im = cap.read()
    cv2.imshow('video test', im)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
    if key == ord(' '):
        cv2.imwrite('vid_result.jpg',im)