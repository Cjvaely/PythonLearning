#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-07 21:17:53
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# hashlib：摘要算法，如MD5，SHA1（哈希算法、散列算法）
# 通过一个函数，把任意长度的数据转换为一个长度固定的数据串
# （通常用16进制的字符串表示）

# 练习：根据用户输入的口令，计算出存储在数据库中的MD5口令：
import hashlib
db = {
    'michael': 'e10adc3949ba59abbe56e057f20f883e',
    'bob': '878ef96e86145580c38c87f0410ad153',
    'alice': '99b1c2188db85afee403b1536010c2c9'
}


def login(user, password):
    md5 = hashlib.md5()
    md5.update(password.encode('utf-8'))
    if db[user] == md5.hexdigest():
        return True
    else:
        return False


# 测试:
assert login('michael', '123456')
assert login('bob', 'abc999')
assert login('alice', 'alice2008')
assert not login('michael', '1234567')
assert not login('bob', '123456')
assert not login('alice', 'Alice2008')
print('ok')

# 练习:根据用户输入的登录名和口令模拟用户注册，计算更安全的MD5：
import hashlib
import random


def get_md5(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()


class User(object):
    def __init__(self, username, password):
        self.username = username
        self.salt = ''.join([chr(random.randint(48, 122)) for i in range(20)])
        self.password = get_md5(password + self.salt)


db = {
    'michael': User('michael', '123456'),
    'bob': User('bob', 'abc999'),
    'alice': User('alice', 'alice2008')
}


def login(username, password):
    user = db[username]
    return user.password == get_md5(password + user.salt)


# 测试:
assert login('michael', '123456')
assert login('bob', 'abc999')
assert login('alice', 'alice2008')
assert not login('michael', '1234567')
assert not login('bob', '123456')
assert not login('alice', 'Alice2008')
print('ok')
