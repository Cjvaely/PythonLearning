#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-25 10:42:32
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : http://www.cnblogs.com/cjvae/
# @Version : $Id$

import asyncio
import threading


# # @asyncio.coroutine把一个generator标记为coroutine（协程）类型
# @asyncio.coroutine
# def hello():
#     print("Hello world!")
#     # 异步调用asyncio.sleep(1):
#     # yield from语法可以让我们方便地调用另一个generator
#     # asyncio.sleep()也是一个coroutine
#     # 线程不会等待asyncio.sleep()，而是直接中断并执行下一个消息循环
#     # 把asyncio.sleep(1)看成是一个耗时1秒的IO操作，在此期间，
#     # 主线程并未等待，而是去执行EventLoop中其他可以执行的coroutine了，因此可以实现并发执行
#     yield from asyncio.sleep(1)
#     print("Hello again!")


# # 从asyncio模块中直接获取EventLoop
# loop = asyncio.get_event_loop()
# # 执行coroutine
# loop.run_until_complete(hello())
# loop.close()


@asyncio.coroutine
def hello():
    print('Hello world! (%s)' % threading.currentThread())
    yield from asyncio.sleep(1)
    print('Hello again! (%s)' % threading.currentThread())


# 从asyncio模块中直接获取EventLoop
loop = asyncio.get_event_loop()
tasks = [hello(), hello()]
loop.run_until_complete(asyncio.wait(tasks))
loop.close()
