#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-30 22:27:39
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : https://github.com/Cjvaely
# @Version : $Id$

# 安装第三方模块
#
# 模块搜索路径
# 默认情况下，Python解释器会搜索当前目录、所有已安装的内置模块和第三方模块，
# 搜索路径存放在sys模块的path变量中：
# >>> import sys
# >>> sys.path

# 如果我们要添加自己的搜索目录，有两种方法：
# 一是直接修改sys.path，添加要搜索的目录：
# >> > import sys
# >> > sys.path.append('/Users/michael/my_py_scripts')
# 这种方法是在运行时修改，运行结束后失效。
#
# 二是：设置环境变量PYTHONPATH，该环境变量的内容会被自动添加到模块搜索路径中。
# 设置方式与设置Path环境变量类似。注意只需要添加你自己的搜索路径，Python自己本身的搜索路径不受影响
