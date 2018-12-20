#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-17 10:54:37
# @Author  : Chen Cjv (cjvaely@foxmail.com)
# @Link    : http://www.cnblogs.com/cjvae/
# @Version : $Id$


class HtmlOutputer(object):
    # 列表维护输出的数据

    def __init__(self):
        self.datas = []

    # 收集数据
    def collect_data(self, data):
        if data is None:
            return
        self.datas.append(data)

    def output_html(self):
        fout = open('output.html', 'w')

        fout.write('<html>')
        fout.write('<body>')
        fout.write('<table>')

        # 生成表格 ascii码
        for data in self.datas:
            # 行的开始标签
            fout.write('<tr>')

            # 输出单元格内容
            fout.write('<td> %s </td>' % data['url'])
            fout.write('<td> %s </td>' % data['title'].encode('utf-8'))
            fout.write('<td> %s </td>' % data['summary'].encode('utf-8'))
            fout.write('<tr>')

            # 行的结束标签
            fout.write('</tr>')

        fout.write('</table>')
        fout.write('</body>')
        fout.write('</html>')

        fout.close()
