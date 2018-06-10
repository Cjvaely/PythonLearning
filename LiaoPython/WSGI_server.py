#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-09 21:35:36
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# server.py:负责启动WSGI服务器，加载application()函数

# 从wsgiref模块导入:
from wsgiref.simple_server import make_server
# 导入我们自己编写的application函数:
from WSGI_hello import application

# 创建一个服务器，IP地址为空，端口是8000，处理函数是application:
httpd = make_server('', 8000, application)
print('Serving HTTP on port 8000...')
# 开始监听HTTP请求:
httpd.serve_forever()

# 打开浏览器，输入http://localhost:8000/，就可以看到结果了
