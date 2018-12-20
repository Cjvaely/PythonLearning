#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-17 10:54:53
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : http://www.cnblogs.com/cjvae/
# @Version : $Id$

from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re


class HtmlParser(object):
    # page_url:当前页面  html_cont:下载的页面内容
    def parse(self, page_url, html_cont, html_encode="utf-8"):
        if page_url is None or html_cont is None:
            return

        soup = BeautifulSoup(html_cont, 'lxml', from_encoding='utf-8')
        new_urls = self._get_new_urls(page_url, soup)
        new_data = self._get_new_data(page_url, soup)
        return new_urls, new_data

    def _get_new_urls(self, page_url, soup):
        # 获取页面中所有的其它的url，我需要的是
        new_urls = set()
        # 找到所有标签
        links = soup.find_all('a', href=re.compile(          # 置顶标题链接
            r"post-read-single.php\?bid=690&type=0&postid=(\d+)$"))
        plinks = soup.find_all('a', href=re.compile(         # 普通标题链接
            r"post-read.php\?bid=690&threadid=(\d+)$"))

        # 合并两个标签
        for plink in plinks:
            links.append(plink)
        for link in links:
            new_url = link['href']
            new_full_url = urljoin(page_url, new_url)
            new_urls.add(new_full_url)
        return new_urls

    def _get_new_data(self, page_url, soup):
        # 存放当前页面url
        res_data = {'url': page_url}

        # 获取标题内容
        title_node = soup.find_all('div', class_="title l limit")
        for title in title_node:
            res_data['title'] = title.get_text()

        # 获取回复数据
        response_node = soup.find('div', class_="post-card").find(
            'p', title="心理咨询师").parent.parent.find(
            'div', class_="body file-read image-click-view").p
        res_data['response'] = response_node.get_text()

        return res_data
