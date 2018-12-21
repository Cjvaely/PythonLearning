#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-21 16:28:37
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : http://www.cnblogs.com/cjvae/
# @Version : $Id$

import os, shutil

# 原始数据集解压目录的路径
original_dataset_dir = '/Users/cjv/Desktop/AI/Keras/dataset/cats_dogs_original_data'
# 保存较小数据集的目录
cat_dir = '/Users/cjv/Desktop/AI/AdrianRosebrockLesson/book/1/knnClassifier/animals/cat'
dog_dir = '/Users/cjv/Desktop/AI/AdrianRosebrockLesson/book/1/knnClassifier/animals/dog'
panda_dir = '/Users/cjv/Desktop/AI/AdrianRosebrockLesson/book/1/knnClassifier/animals/panda'

# 将前 1000 张猫的图像复制到 cat_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(cat_dir, fname)
    shutil.copyfile(src, dst)

# 将前 1000 张狗的图像复制到 dog_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(dog_dir, fname)
    shutil.copyfile(src, dst)

# 将前 1000 张熊猫的图像复制到 panda_dir
