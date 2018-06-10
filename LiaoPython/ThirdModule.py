#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-09 14:19:44
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# 常用的第三方模块之Pillow
# PIL：Python Imaging Library，已经是Python平台事实上的图像处理标准库了。
# PIL功能非常强大，但API却非常简单易用
# 在PIL的基础上创建了兼容的版本，名字叫Pillow，支持最新Python 3.x，又加入了许多新特性，因此，我们可以直接安装使用Pillow

# 操作图像:缩放
from PIL import Image
from PIL import ImageFilter
from PIL import ImageDraw
from PIL import ImageFont
import random


# # 打开一个jpg图像文件，注意是当前路径:
# im = Image.open('C:/Users/cjvae/Pictures/python.png')
# # 获得图像尺寸:
# w, h = im.size
# print('Original image size: %sx%s' % (w, h))
# # 缩放到50%:
# im.thumbnail((w // 2, h // 2))
# print('Resize image to: %sx%s' % (w // 2, h // 2))
# # 把缩放后的图像用jpeg格式保存:
# im.save('C:/Users/cjvae/Pictures/thumbnail.png', 'png')


# # 操作图像:模糊效果
# # 打开一个jpg图像文件，注意是当前路径:
# im = Image.open('C:/Users/cjvae/Pictures/dog.jpg')
# # 应用模糊滤镜:
# im2 = im.filter(ImageFilter.BLUR)
# im2.save('C:/Users/cjvae/Pictures/blur.jpg', 'jpeg')

# 操作图像:生成字母验证码图片


# 随机字母:
def rndChar():
    return chr(random.randint(65, 90))

# 随机颜色1:


def rndColor():
    return (random.randint(64, 255), random.randint(64, 255),
            random.randint(64, 255))

# 随机颜色2:


def rndColor2():
    return (random.randint(32, 127), random.randint(32, 127),
            random.randint(32, 127))


# 240 x 60:
width = 60 * 4
height = 60
image = Image.new('RGB', (width, height), (255, 255, 255))
# 创建Font对象:
font = ImageFont.truetype('arial.ttf', 36)
# 创建Draw对象:
draw = ImageDraw.Draw(image)
# 填充每个像素:
for x in range(width):
    for y in range(height):
        draw.point((x, y), fill=rndColor())
# 输出文字:
for t in range(4):
    draw.text((60 * t + 10, 10), rndChar(), font=font, fill=rndColor2())
# 模糊:
image = image.filter(ImageFilter.BLUR)
image.save('C:/Users/cjvae/Pictures/code.jpg', 'jpeg')
