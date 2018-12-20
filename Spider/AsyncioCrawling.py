#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-23 16:49:57
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : http://www.cnblogs.com/cjvae/
# @Version : 异步爬取

"""
在单线程里使用异步计算, 下载网页的时候和处理网页的时候是不连续的
,更有效利用了等待下载的这段时间。
在Python的功能间切换着执行
切换的点用 await 来标记, 能够异步的功能用 async标记
"""
import time
import asyncio
import requests
import aiohttp


# # 普通进程的执行
# def job(t):
#     print('Start job:', t)
#     time.sleep(t)
#     print('Job ', t, 'takes', t, 's')


# def main():
#     [job(t) for t in range(1, 3)]


# t1 = time.time()
# main()
# print('No async total time:', time.time() - t1)

# print('\n')


# # 加入异步
# async def job1(t):  		# async形式的功能
#     print('Start job ', t)
#     await asyncio.sleep(t)  # 等待的t秒时，切换到其他任务
#     print('Job ', t, ' takes ', t, ' s')


# async def main1(loop):
#     # 创建任务, 但是不执行
#     tasks = [
#         loop.create_task(job1(t)) for t in range(1, 3)
#     ]
#     # 执行并等待所有任务完成
#     await asyncio.wait(tasks)
# t2 = time.time()
# # 建立 loop
# loop = asyncio.get_event_loop()
# # 执行loop
# loop.run_until_complete(main1(loop))
# loop.close()
# print('Async total time:', time.time() - t2)

# print('\n')

# 普通请求网页
URL = 'https://morvanzhou.github.io/'


def normal():
    for i in range(2):
        r = requests.get(URL)
        url = r.url
        print(url)


t3 = time.time()
normal()
print('Normal total time:', time.time() - t3)


# 异步请求网页
async def NotNormal(session):
    # 等待并切换
    response = await session.get(URL)
    return str(response.url)


async def main2(loop):
    async with aiohttp.ClientSession() as session:
        tasks = [loop.create_task(NotNormal(session)) for _ in range(2)]
        finished, unfinished = await asyncio.wait(tasks)
        # 获取所有结果
        all_results = [r.result() for r in finished]
        print(all_results)
t4 = time.time()
loop = asyncio.get_event_loop()
loop.run_until_complete(main2(loop))
loop.close()
print('Async total time:', time.time() - t4)
