#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-09 21:28:26
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# WSGI接口：只要求Web开发者实现一个函数，就可以响应HTTP请求


def application(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/html')])
    return [b'<h1>Hello, web!</h1>']

# environ：一个包含所有HTTP请求信息的dict对象；
# start_response：一个发送HTTP响应的函数。
# start_response()函数接收两个参数，一个是HTTP响应码，
# 一个是一组list表示的HTTP Header，每个Header用一个包含两个str的tuple表示
# 通常情况下，都应该把Content-Type头发送给浏览器
# 函数的返回值b'<h1>Hello, web!</h1>'将作为HTTP响应的Body发送给浏览器。

# Python内置了一个WSGI服务器，这个模块叫wsgiref
