#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-17 10:52:20
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : http://www.cnblogs.com/cjvae/
# @Version : $I

# python 3.x
from urllib import request
# python 2.x
# import urllib2


class HtmlDownloader(object):  # 下载某个页面
    def download(self, url):
        if url is None:
            return None

        response = request.Request.urlopen(url)
        content = response.read()

        if response.getcode() != 200:  # 请求失败
            return None

        return (str(content), 'utf-8')
