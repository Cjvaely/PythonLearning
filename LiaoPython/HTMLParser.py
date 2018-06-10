#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-09 11:01:41
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# HTMLParser：解析HTML的模板：
#
# tag是的html标签，attrs是 (属性，值)元组(tuple)的列表(list).
# 如一个标签为：
# <input type="hidden" name="NXX" id="IDXX" value="VXX" />
# 那么它的attrs列表为
# [('type', 'hidden'), ('name', 'NXX'), ('id', 'IDXX'), ('value', 'VXX')]

# 解析时碰到<***>，自动调用handle_starttag()；碰到</***>，自动调用handle_endtag()
# 每一个标签，无论<> 还是</>，均会调用handle_data()
# html中第一行、第二行分别为<html>和<head>，后面无具体数据，只有回车换行，
# 所用调用handle_data()，打印结果为换行；</html></head>同理


# from html.parser import HTMLParser
# from html.entities import name2codepoint


# class MyHTMLParser(HTMLParser):

# 对开始标签的处理方法,<div id="main">，参数tag指的是div，attrs指的是一个（name,Value)的列表
#     def handle_starttag(self, tag, attrs):
#         print('starttag:<%s>' % tag)

# 对结束标签的处理方法。例如</div>，参数tag指的是div
#     def handle_endtag(self, tag):
#         print('endtag:</%s>' % tag)

# 处理开始标签和结束标签
#     def handle_startendtag(self, tag, attrs):
#         print('startendtag:<%s/>' % tag)

# 对标签之间的数据的处理方法。<tag>test</tag>,data指的是“test”
#     def handle_data(self, data):
#         print('data:' + data)

# 对HTML中注释的处理方法。
#     def handle_comment(self, data):
#         print('comment:<!--', data, '-->')

# 处理一些特殊字符，以 & 开头的，比如 & nbsp
#     def handle_entityref(self, name):
#         print('entityref: &%s;' % name)

# 处理特殊字符串，就是以&  # 开头的，一般是内码表示的字符
#     def handle_charref(self, name):
#         print('charref: &#%s;' % name)


# parser = MyHTMLParser()
# parser.feed('''<html>
# <head></head>
# <body>
# <!-- test html parser -->
#     <p>Some <a href=\"#\">html</a> HTML&nbsp;tutorial...<br>END</p>
# </body></html>''')


# 练习:一个网页，例如https://www.python.org/events/python-events/，用浏览器查看源码并复制，然后尝试解析一下HTML，输出Python官网发布的会议时间、名称和地点。
from html.parser import HTMLParser
from urllib import request


class MyHTMLParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.html_dict = {'time': None,
                          'event-title': None, 'event-location': None}

    def handle_starttag(self, tag, attrs):
        if tag == 'time':
            self.html_dict['time'] = True
        # attrs的形式
        elif tag == 'span' and attrs.__contains__(('class', 'event-location')):
            self.html_dict['event-location'] = True
        elif tag == 'h3' and attrs.__contains__(('class', 'event-title')):
            self.html_dict['event-title'] = True

    def handle_data(self, data):
        if self.html_dict['time']:
            print('Time: %s\t' % data)
            self.html_dict['time'] = None
        elif self.html_dict['event-title']:
            print('Title: %s\t' % data)
            self.html_dict['event-title'] = None
        elif self.html_dict['event-location']:
            print('Location: %s\t' % data)
            self.html_dict['event-location'] = None
            print('\n')


parser = MyHTMLParser()
with request.urlopen('http://www.python.org/events/python-events/') as f:
    data = f.read().decode('utf-8')

parser.feed(data)
