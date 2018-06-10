#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-06 21:55:38
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# Server Process
# 服务进程:启动Queue，把Queue注册到网络上，然后往Queue里面写入任务
# task_master.py

import random
import time
import queue
from multiprocessing.managers import BaseManager
from multiprocessing import freeze_support

# 发送任务的队列:
task_queue = queue.Queue()
# 接收结果的队列:
result_queue = queue.Queue()

# 从BaseManager继承的QueueManager:


class QueueManager(BaseManager):
    pass


# 把两个Queue都注册到网络上, callable参数关联了Queue对象:
'''仅适用Linux Windows下callable不能使用lambda表达式赋值
QueueManager.register('get_task_queue', callable=lambda: task_queue)
QueueManager.register('get_result_queue', callable=lambda: result_queue)
'''


def return_task_queue():
    global task_queue
    return task_queue


def return_result_queue():
    global result_queue
    return result_queue


def runf():
    QueueManager.register('get_task_queue', callable=return_task_queue)
    QueueManager.register('get_result_queue', callable=return_result_queue)
    # 绑定端口5000，设置验证密码'abc'
    manager = QueueManager(address=('127.0.0.1', 5000), authkey=b'abc')
    # Linux下address留空等于本机 Windows下不能留空 127.0.0.0即本机的地址
    # 启动Queue
    manager.start()
    # 通过网络获取Queue对象
    task = manager.get_task_queue()
    result = manager.get_result_queue()
    # 开启示例任务
    for i in range(10):
        n = random.randint(0, 10000)
        print('Put task %d to run...' % n)
        task.put(n)
    # 读取任务结果
    print('Try to get results...')
    for i in range(10):
        r = result.get(timeout=10)
        print('Results: %s' % r)
    manager.shutdown()
    print('master has been shoutdown')


if __name__ == '__main__':
    freeze_support()
    runf()
