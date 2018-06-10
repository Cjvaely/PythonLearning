#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-09 15:06:53
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# requests:Python第三方库，处理URL资源特别方便

# 通过GET访问一个页面，只需要几行代码：
import requests
r = requests.get('https://www.douban.com/')  # 豆瓣首页
print(r.status_code)
print(r.text)

# 无论响应是文本还是二进制内容，我们都可以用content属性获得bytes对象：
print(r.content)

# 对于特定类型的响应，例如JSON，可以直接获取：
r = requests.get(
    'https://query.yahooapis.com/v1/public/yql?q=select%20*%20from%20weather.forecast%20where%20woeid%20%3D%202151330&format=json')
print(r.json())

# 需要传入HTTP Header时，我们传入一个dict作为headers参数：
r = requests.get('https://www.douban.com/', headers={
                 'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 11_0 like Mac OS X) AppleWebKit'})
print(r.text)

# 要发送POST请求，只需要把get()方法变成post()，然后传入data参数作为POST请求的数据：
r = requests.post('https://accounts.douban.com/login',
                  data={'form_email': 'abc@example.com', 'form_password': '123456'})

# 上传文件需要更复杂的编码格式，但是requests把它简化成files参数：
upload_files = {'file': open('report.xls', 'rb')}
r = requests.post('https://www.douban.com/', files=upload_files)

# 获取响应头：
print(r.headers)

# 获取指定的Cookie:
r.cookies['ts']

# 请求中传入Cookie，只需准备一个dict传入cookies参数：
r = requests.get(url, timeout=2.5)  # 2.5秒后超时
