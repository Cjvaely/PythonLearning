#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-22 15:35:02
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : http://www.cnblogs.com/cjvae/
# @Version : 多进程爬取

from urllib.request import urlopen, urljoin
from bs4 import BeautifulSoup
import multiprocessing as mp
import re
import time

base_url = 'https://morvanzhou.github.io/'

# DON'T OVER CRAWL THE WEBSITE OR YOU MAY NEVER VISIT AGAIN
restricted_crawl = True

# 定义爬取的方法


def crawl(url):
    response = urlopen(url)
    time.sleep(0.1)             # slightly delay for downloading
    return response.read().decode()


def parse(html):
    soup = BeautifulSoup(html, 'lxml')
    urls = soup.find_all('a', {"href": re.compile('^/.+?/$')})
    # 当前页面的标题
    title = soup.find('h1').get_text().strip()
    # 当前页面的所有url
    page_urls = set([urljoin(base_url, url['href'])
                     for url in urls])   # remove duplication
    # 当前页面的url
    url = soup.find('meta', {'property': "og:url"})['content']
    return title, page_urls, url


if __name__ == '__main__':
    # 未爬取的url
    unseen = set([base_url, ])
    # 已经爬取的url
    seen = set()

    # number strongly affected
    pool = mp.Pool(4)
    # 记录爬取的个数和时间
    count, t1 = 1, time.time()
    # still get some url to visit
    while len(unseen) != 0:
        # 限制爬取数量
        if restricted_crawl and len(seen) > 30:
            break
        print('\nDistributed Crawling...')
        # pool.apply_async方法只能传入一个参数，分配给进程池使用
        crawl_jobs = [pool.apply_async(crawl, args=(url,)) for url in unseen]
        # request connection
        # crawl_jobs是爬取的网页内容的集合
        htmls = [j.get() for j in crawl_jobs]
        # 移除空网页
        htmls = [h for h in htmls if h is not None]

        print('\nDistributed Parsing...')
        parse_jobs = [pool.apply_async(parse, args=(html,)) for html in htmls]
        # parse html
        results = [j.get() for j in parse_jobs]

        print('\nAnalysing...')
        seen.update(unseen)
        unseen.clear()

        for title, page_urls, url in results:
            print(count, title, url)
            count += 1
            unseen.update(page_urls - seen)

    print('Total time: %.1f s' % (time.time() - t1, ))
