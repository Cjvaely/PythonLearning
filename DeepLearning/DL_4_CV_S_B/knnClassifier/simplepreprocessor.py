#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-21 02:17:37
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : http://www.cnblogs.com/cjvae/
# @Version : $Id$

import cv2

class SimplePreprocessor:
	# 使用像素区域关系进行重采样。它可能是图像抽取的首选方法，因为它会产生无云纹理的结果
     def __init__(self, width, height, inter=cv2.INTER_AREA):
          # 存储调整大小时使用的目标图像宽度，高度和插值方法
          self.width = width
          self.height = height
          self.inter = inter
          
     def preprocess(self, image):
          # 将图像调整为固定大小，忽略纵横比
          return cv2.resize(image, (self.width, self.height),
               interpolation=self.inter)

if __name__ == '__main__':

     s = SimplePreprocessor(32, 32)
     img = cv2.imread('/Users/cjv/Desktop/AI/AdrianRosebrockLesson/book/1/knnClassifier/animals/dog/dog.16.jpg')
     cv2.imshow('src', img)
     cv2.imshow("resize", s.preprocess(img))
     print(img.size)
     cv2.waitKey(0)
