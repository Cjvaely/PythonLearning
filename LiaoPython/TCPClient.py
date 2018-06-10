#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-09 16:10:39
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# TCP网络编程:
import socket

# # 创建一个socket,AF_INET->IPv4  AF_INET6->IPv6, SOCK_STREAM指定使用面向流的TCP协议
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# # 建立连接,作为服务器，提供什么样的服务，
# 端口号就必须固定下来。由于我们想要访问网页，因此新浪提供网页服务的服务器必须把端口号固定在80端口，因为80端口是Web服务的标准端口
# s.connect(('www.sina.com', 80))
# # 发送数据
# s.send(b'GET / HTTP/1.1\r\nHost:www.sina.com.cn\r\nConnection:close\r\n\r\n')
# # 接收数据:一次最多接收指定的字节数，因此，在一个while循环中反复接收，直到recv()返回空数据，表示接收完毕，退出循环
# buffer = []
# while True:
#     # 每次最多收1K字节
#     d = s.recv(1024)
#     if d:
#         buffer.append(d)
#     else:
#         break
# data = b''.join(buffer)
# # 接收完数据后，调用close()方法关闭Socket
# s.close()
# # 接收到的数据包括HTTP头和网页本身，我们只需要把HTTP头和网页分离一下，把HTTP头打印出来，网页内容保存到文件：
# header, html = data.split(b'\r\n\r\n', 1)
# print(header.decode('utf-8'))
# # 把接受的数据写入文件：
# with open('sina.html', 'wb') as f:
#     f.write(html)


# TCPClient
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 建立连接:
s.connect(('127.0.0.1', 9999))
# 接收欢迎消息:
print(s.recv(1024).decode('utf-8'))
for data in [b'Michael', b'Tracy', b'Sarah']:
    # 发送数据:
    s.send(data)
    print(s.recv(1024).decode('utf-8'))
s.send(b'exit')
s.close()
