#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-21 16:29:58
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : http://www.cnblogs.com/cjvae/
# @Version : Test the methord: post and get

import requests
import webbrowser

# test get
param = {'wd': '许嵩'}
rg = requests.get('http://www.baidu.com/s', params=param)
print(rg.url)
webbrowser.open(rg.url)

# test post
data = {"firstname": 'Cjv', "lastname": "Chan"}
rp = requests.post('http://pythonscraping.com/pages/files/processing.php',
                   data=data)
print(rp.text)
