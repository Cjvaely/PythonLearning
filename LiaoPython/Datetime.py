#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-07 16:36:51
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# datetime:处理日期和时间的标准库

# datetime.now()返回当前日期和时间，其类型是datetime

# timestamp:当前时间就是相对于epoch time(1970年1月1日 00:00:00 UTC+00:00时区的时刻)的秒数
# timestamp = 0 = 1970-1-1 00:00:00 UTC+0:00

# datetime转换为timestamp:dt.timestamp()
# timestamp转换为datetime:datetime.fromtimestamp(t)

# str转换为datetime:datetime.strptime('timestr','format')
# datetime转换为str:now.strftime()

# 练习：假设你获取了用户输入的日期和时间如2015-1-21 9:01:30，
# 以及一个时区信息如UTC+5:00，均是str，请编写一个函数将其转换为timestamp：
import re
from datetime import datetime, timezone, timedelta


def to_timestamp(dt_str, tz_str):
    st = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    ts = st.timestamp()
    ls = re.match(r'UTC(\+|\-)([0-9]?[0-9]+):(00)$', tz_str)
    if ls:
        if ls.group(1) == '+':
            stamp = ts + (8 - int(ls.group(2))) * 3600
        else:
            stamp = ts + (8 + int(ls.group(2))) * 3600
    return stamp

# 测试:


t1 = to_timestamp('2015-6-1 08:10:30', 'UTC+7:00')
assert t1 == 1433121030.0, t1

t2 = to_timestamp('2015-5-31 16:10:30', 'UTC-09:00')
assert t2 == 1433121030.0, t2

print('ok')
