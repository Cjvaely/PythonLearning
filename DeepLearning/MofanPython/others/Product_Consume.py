#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-24 21:04:16
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : http://www.cnblogs.com/cjvae/
# @Version : $Id$

# python producer&consumer by coroutine(协程)


# consumer函数是一个generator
def consumer():
    r = 'k'  # 定义一个空字符串
    while True:  # 设定一个循环
        # 如果不调用consumer的send方法传入其参数给n，n将为None
        n = yield r
        if not n:  # 如果满足条件，表示方法外并未调用send
            return  # 执行return，退出方法，返回空值
        print('[CONSUMER] Consuming %s...' % n)
        r = '200 OK'


def produce(c):
    x = c.send(None)  # 启动生成器,相当于调用了next(c)
    print("This is qidong:", x)
    n = 0
    while n < 5:
        n = n + 1
        print('[PRODUCER] Producing %s...' % n)
        # 生产了东西，通过c.send(n)切换到consumer执行
        r = c.send(n)
        # consumer通过yield拿到消息，处理，又通过yield把结果传回
        print('[PRODUCER] Consumer return: %s' % r)
        # produce拿到consumer处理的结果，继续生产下一条消息
    # produce决定不生产了，通过c.close()关闭consumer，整个过程结束
    c.close()


c = consumer()
produce(c)
