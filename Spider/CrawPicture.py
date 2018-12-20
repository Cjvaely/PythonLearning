#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-21 17:10:29
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : http://www.cnblogs.com/cjvae/
# @Version : 爬取国家地理杂志图片

from bs4 import BeautifulSoup
import requests
import os
URL = "http://www.nationalgeographic.com.cn/animals/"
os.makedirs('./img/', exist_ok=True)

html = requests.get(URL).text
soup = BeautifulSoup(html, 'lxml')
img_ul = soup.find_all('ul', {"class": "img_list"})

for ul in img_ul:
    imgs = ul.find_all('img')
    for img in imgs:
        url = img['src']
        r = requests.get(url, stream=True)
        image_name = url.split('/')[-1]
        with open('./img/%s' % image_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=128):
                f.write(chunk)
        print('Saved %s' % image_name)
