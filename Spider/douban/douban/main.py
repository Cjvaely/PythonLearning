#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-23 21:08:23
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : http://www.cnblogs.com/cjvae/
# @Version : 编写运行脚本

from scrapy import cmdline
cmdline.execute('scrapy crawl douban_spider'.split())
