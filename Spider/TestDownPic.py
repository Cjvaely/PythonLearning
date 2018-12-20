#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-21 16:43:06
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : http://www.cnblogs.com/cjvae/
# @Version : 下载图片

# 下载功能 urlretrieve
from urllib.request import urlretrieve
# 使用requests下载, 下载大文件效率高

import requests
IMAGE_URL = ("https://morvanzhou.github.io/static" +
             "/img/description/learning_step_flowchart.png")
urlretrieve(IMAGE_URL, './img/img1.png')
print("urlretrieve download a picture!")

# 使用requests下载, 下载大文件效率高
# requests 能让你下一点, 保存一点
# 使用 r.iter_content(chunk_size) 来控制每个 chunk 的大小
# 然后在文件中写入这个 chunk 大小的数据.
r = requests.get(IMAGE_URL)
with open('./img/img2.png', 'wb') as f:
    f.write(r.content)
print("requests download a picture!")

r1 = requests.get(IMAGE_URL, stream=True)

with open('./img/img3.png', 'wb') as f:
    for chunk in r1.iter_content(chunk_size=32):
        f.write(chunk)
print("requests.chunk download a picture!")
