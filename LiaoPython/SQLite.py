#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-09 17:16:29
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# SQLite
# 练习:请编写函数，在Sqlite中根据分数段查找指定的名字：

import os
import sqlite3

db_file = os.path.join(os.path.dirname(__file__), 'test.db')
if os.path.isfile(db_file):
    os.remove(db_file)

# 初始数据:
conn = sqlite3.connect(db_file)
cursor = conn.cursor()
cursor.execute(
    'create table user(id varchar(20) primary key,name varchar(20),score int)')
cursor.execute(r"insert into user values ('A-001', 'Adam', 95)")
cursor.execute(r"insert into user values ('A-002', 'Bart', 62)")
cursor.execute(r"insert into user values ('A-003', 'Lisa', 78)")
cursor.close()
conn.commit()
conn.close()


def get_score_in(low, high):
    ''' 返回指定分数区间的名字，按分数从低到高排序 '''
    # lt用来存符合分数区间的组成
    lt = []
    # name用来存返回的名字
    name = []
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    results = cursor.execute(r'select * from user')
    # 从数据库表中取出的结果，[(), (), ()...]
    all_users = results.fetchall()
    for user in all_users:
        if low <= user[2] <= high:
            lt.append(user)
    # 排序
    lt = sorted(lt, key=lambda lt: lt[2])
    for u in lt:
        name.append(u[1])
    conn.close()
    cursor.close()
    return name
    # 测试:


assert get_score_in(80, 95) == ['Adam'], get_score_in(80, 95)
assert get_score_in(60, 80) == ['Bart', 'Lisa'], get_score_in(60, 80)
assert get_score_in(60, 100) == ['Bart', 'Lisa', 'Adam'], get_score_in(60, 100)

print('Pass')
