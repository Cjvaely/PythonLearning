#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-07 20:41:31
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# struct:解决bytes和其他二进制数据类型的转换
# struct的pack函数把任意数据类型变成bytes
# >>> import struct
# >>> struct.pack('>I', 10240099)
# b'\x00\x9c@c'
#
# '>I'的意思是：
# >表示字节顺序是big-endian，也就是网络序，I表示4字节无符号整数。
# 后面的参数个数要和处理指令一致
#
# unpack把bytes变成相应的数据类型
# >>> struct.unpack('>IH', b'\xf0\xf0\xf0\xf0\x80\x80')
# (4042322160, 32896)
#
# 请编写一个bmpinfo.py，可以检查任意文件是否是位图文件，如果是，打印出图片大小和颜色数。
import base64
import struct

bmp_data = base64.b64decode('Qk1oAgAAAAAAADYAAAAoAAAAHAAAAAoAAAABABAAAAAAADICAAASCwAAEgsAAAAAAAAAAAAA/3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9/AHwAfAB8AHwAfAB8AHwAfP9//3//fwB8AHwAfAB8/3//f/9/AHwAfAB8AHz/f/9//3//f/9//38AfAB8AHwAfAB8AHwAfAB8AHz/f/9//38AfAB8/3//f/9//3//fwB8AHz/f/9//3//f/9//3//f/9/AHwAfP9//3//f/9/AHwAfP9//3//fwB8AHz/f/9//3//f/9/AHwAfP9//3//f/9//3//f/9//38AfAB8AHwAfAB8AHwAfP9//3//f/9/AHwAfP9//3//f/9//38AfAB8/3//f/9//3//f/9//3//fwB8AHwAfAB8AHwAfAB8/3//f/9//38AfAB8/3//f/9//3//fwB8AHz/f/9//3//f/9//3//f/9/AHwAfP9//3//f/9/AHwAfP9//3//fwB8AHz/f/9/AHz/f/9/AHwAfP9//38AfP9//3//f/9/AHwAfAB8AHwAfAB8AHwAfAB8/3//f/9/AHwAfP9//38AfAB8AHwAfAB8AHwAfAB8/3//f/9//38AfAB8AHwAfAB8AHwAfAB8/3//f/9/AHwAfAB8AHz/fwB8AHwAfAB8AHwAfAB8AHz/f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//38AAA==')


def bmp_info(data):
    b64 = struct.unpack('<ccIIIIIIHH', data[:30])
    if b64[0] + b64[1] == b'BM':
        print('The size of bmp:%s\n' % b64[2])
        print('The amount of bmp_color:%s' % b64[9])
        return {
            'width': b64[6],
            'height': b64[7],
            'color': b64[9]
        }


# 测试
bi = bmp_info(bmp_data)
assert bi['width'] == 28
assert bi['height'] == 10
assert bi['color'] == 16
print('ok')
