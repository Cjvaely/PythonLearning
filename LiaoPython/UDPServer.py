#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-09 16:43:25
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# UDP:不需要建立连接，只需要知道对方的IP地址和端口号，就可以直接发数据包。但是，能不能到达就不知道了
# UDPServer
import socket

# SOCK_DGRAM指定了这个Socket的类型是UDP
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# 绑定端口:
s.bind(('127.0.0.1', 9999))
print('Bind UDP on 9999...')
while True:
    # 接收数据:recvfrom()方法返回数据和客户端的地址与端口
    data, addr = s.recvfrom(1024)
    print('Received from %s:%s.' % addr)
    # 直接调用sendto()就可以把数据用UDP发给客户端
    s.sendto(b'Hello, %s!' % data, addr)
