#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-08 20:16:11
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# XML:操作XML有两种方法：DOM和SAX
# DOM会把整个XML读入内存，解析为树，因此占用内存大，解析慢，优点是可以任意遍历树的节点
# SAX是流模式，边读边解析，占用内存小，解析快，缺点是我们需要自己处理事件
# 正常情况下，优先考虑SAX
# 使用SAX解析XML非常简洁，通常我们关心的事件是start_element，end_element和char_data

# 举个例子，当SAX解析器读到一个节点时：
# <a href="/">python</a>

# start_element事件：读取<a href = "/">时
# char_data事件：读取python时
# end_element事件：读取</a>时
from xml.parsers.expat import ParserCreate


# class DefaultSaxHandler(object):
#     def start_element(self, name, attrs):
#         print('sex:start_element: %s, attrs: %s' % (name, str(attrs)))

#     def end_element(self, name):
#         print('sax: end_element: % s' % name)

#     def char_data(self, text):
#         print('sax:char_data: %s' % text)


# xml = r'''<?xml version="1.0"?>
# <ol>
#     <li><a href="/python">Python</a></li>
#     <li><a href="/ruby">Ruby</a></li>
# </ol>
# '''

# handler = DefaultSaxHandler()
# parser = ParserCreate()  # 创建解析器实例
# # 赋予相应的处理方法(解析器的正常工作还依赖于我们自己定义的处理函数)
# parser.StartElementHandler = handler.start_element
# parser.EndElementHandler = handler.end_element
# parser.CharacterDataHandler = handler.char_data
# parser.Parse(xml)
# parser.StartElementHandler、parser.EndElementHandler、
# parser.CharacterDataHandler这三个方法的名称是固定的，
# 而后面赋予的处理方法由我们自己定义，这些方法所做的事情
# 要和parser.StartElementHandler、parser.EndElementHandler、
# parser.CharacterDataHandler这三者的名称相匹配。


# 练习：请利用SAX编写程序解析Yahoo的XML格式的天气预报，获取天气预报：
from urllib import request

parse_dict = {}


class DefaultSaxHandler(object):
    def start_element(self, name, attrs):
        if name == 'yweather:location':
            parse_dict['city'] = attrs['city']


def parseXml(xml_str):
    print(xml_str + '\n')
    handler = DefaultSaxHandler()
    parser = ParserCreate()
    parser.StartElementHandler = handler.start_element
    parser.Parse(xml_str)
    return parse_dict


# 测试:


URL = 'https://query.yahooapis.com/v1/public/yql?q=select%20*%20from%20weather.forecast%20where%20woeid%20%3D%202151330&format=xml'

with request.urlopen(URL, timeout=4) as f:
    data = f.read()

result = parseXml(data.decode('utf-8'))
assert result['city'] == 'Beijing'
print('ok')
