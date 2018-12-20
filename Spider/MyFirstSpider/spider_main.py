#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-17 09:52:38
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : http://www.cnblogs.com/cjvae/
# @Version : $Id$

from MyFirstSpider import url_manager
from MyFirstSpider import html_downloader, html_parser, html_outputer

'''
爬取url为： https://bbs.pku.edu.cn/v2/thread.php?bid=690
爬标题和心理咨询师的回复
'''


class SpiderMain(object):
    def __init__(self):
        # URL管理器
        self.urls = url_manager.UrlManager()
        # 下载器
        self.downloader = html_downloader.HtmlDownloader()
        # 解析器
        self.parser = html_parser.HtmlParser()
        # 输出器
        self.outputer = html_outputer.HtmlOutputer()

    def craw(self, root_url):
        count = 1                           # 记录当前爬取的是第几个url
        self.urls.add_new_url(root_url)     # 添加入口URL
        while self.urls.has_new_url():      # 管理器中有在爬取的url，即new_urls非空
            try:
                new_url = self.urls.get_new_url()  # 获取一个待爬取的url
                print('Craw %d : %s' % (count, new_url))
                html_cont = self.downloader.download(new_url)   # 下载url对应的页面

                # 解析当前在爬取的url页面，查找需要的数据（new_data）和链接（new_urls）
                new_urls, new_data = self.parser.parse(
                    new_url, html_cont, "utf-8")

                # 批量添加当前页面获取的url，以及数据收集
                self.urls.add_new_urls(new_urls)
                self.outputer.collect_data(new_data)

                # if count == 1000:       # 爬取1000条url
                # break

                # count = count + 1
            except Exception as e:
                print("Craw failed!\n" + str(e))

        self.outputer.output_html()     # 打印数据


if __name__ == "__main__":
    root_url = "https://bbs.pku.edu.cn/v2/thread.php?bid=690"
    obj_spider = SpiderMain()
    obj_spider.craw(root_url)
