#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-26 17:16:19
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : http://www.cnblogs.com/cjvae/
# @Version : $Id$

from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'


if __name__ == '__main__':
	app.run()
