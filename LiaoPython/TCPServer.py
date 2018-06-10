#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-09 16:30:05
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# TCPServer:服务器端
import socket
import threading
import time


def tcplink(sock, addr):
    print('Accept new connection from %s:%s...' % addr)
    # 连接建立后，服务器首先发一条欢迎消息
    sock.send(b'Welcome!')
    # 等待客户端数据，并加上Hello再发送给客户端。如果客户端发送了exit字符串，就直接关闭连接
    while True:
        data = sock.recv(1024)
        time.sleep(1)
        if not data or data.decode('utf-8') == 'exit':
            break
        sock.send(('Hello, %s!' % data.decode('utf-8')).encode('utf-8'))
    sock.close()
    print('Connection from %s:%s closed.' % addr)


# 创建一个基于IPv4和TCP协议的Socket：
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 绑定监听的地址和端口，0.0.0.0绑定到所有的网络地址，127.0.0.1绑定到本机地址，客户端必须同时在本机运行才能连接，也就是说，外部的计算机无法连接进来
# 监听端口:
s.bind(('127.0.0.1', 9999))
# 调用listen()方法开始监听端口，传入的参数指定等待连接的最大数量：
s.listen(5)
print('Waiting for connection...')
# 永久循环来接受来自客户端的连接，accept()会等待并返回一个客户端的连接:
while True:
    # 接受一个新连接:
    sock, addr = s.accept()
    # 创建新线程来处理TCP连接:
    t = threading.Thread(target=tcplink, args=(sock, addr))
    t.start()
