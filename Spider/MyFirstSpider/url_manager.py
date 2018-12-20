#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-17 10:55:03
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : http://www.cnblogs.com/cjvae/
# @Version : $Id$


class UrlManager(object):
    def __init__(self, arg):
        self.new_urls = set()
        self.odd_urls = set()

    def add_new_url(self, url):
        if url is None:
            return
        if url not in self.new_urls and url not in self.odd_urls:
            self.new_urls.add(url)

    def add_new_urls(self, urls):  	# 批量添加url
        if urls is None or len(urls) == 0:
            return
        for url in urls:
            self.add_new_url(url)  # 调用添加单个url的方法

    def has_new_url(self):  	  # 判断是否有新的待爬取的URL
        return len(self.new_urls) != 0

    def get_new_url(self):  # 获取一个爬取的url
        new_url = self.new_urls.pop()
        self.odd_urls.add(new_url)
        return new_url
