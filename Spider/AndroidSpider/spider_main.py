#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-17 10:31:17
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : http://www.cnblogs.com/cjvae/
# @Version : $Id$

from AndroidSpider import url_manager
from AndroidSpider import html_downloader, html_parser, html_output

'''
爬取百度百科 Android 关键词相关词及简介并输出为一个HTML tab网页
Extra module:
BeautifulSoup
'''


class SpiderMain(object):
    def __init__(self):
        self.urls = url_manager.UrlManager()                # url管理器
        self.downloader = html_downloader.HtmlDownLoader()  # 下载器
        self.parser = html_parser.HtmlParser()              # 解析器
        self.out_put = html_output.HtmlOutput()             # 输出器

    def craw(self, root_url):
        count = 1
        self.urls.add_new_url(root_url)
        while self.urls.has_new_url():
            try:
                new_url = self.urls.get_new_url()
                print("craw %d : %s" % (count, new_url))
                headers = {
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.100 Safari/537.36"
                }
                html_content = self.downloader.download(
                    new_url, retry_count=2, headers=headers)
                new_urls, new_data = self.parser.parse(
                    new_url, html_content, "utf-8")
                self.urls.add_new_urls(new_urls)
                self.out_put.collect_data(new_data)
                if count >= 30:
                    break
                count = count + 1
            except Exception as e:
                print("craw failed!\n"+str(e))
        self.out_put.output_html()


if __name__ == "__main__":
    rootUrl = "http://baike.baidu.com/item/Android" // 要爬取的URL入口
    objSpider = SpiderMain()
    objSpider.craw(rootUrl)
