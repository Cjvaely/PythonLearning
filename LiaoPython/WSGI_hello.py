#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-09 21:33:57
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# hello.py:实现Web应用程序的WSGI处理函数


def application(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/html')])
    return [b'<h1>Hello, web!</h1>']
