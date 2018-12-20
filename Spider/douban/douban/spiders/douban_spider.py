# -*- coding: utf-8 -*-
# 爬虫目标:https://movie.douban.com/top250
import scrapy
from douban.items import DoubanItem


class DoubanSpiderSpider(scrapy.Spider):
    # 爬虫名
    name = 'douban_spider'
    # 允许的域名
    allowed_domains = ['movie.douban.com']
    # 入口URL， 扔到调度器
    start_urls = ['https://movie.douban.com/top250']

    # 默认解析方法
    def parse(self, response):
        movie_list = response.xpath(
            "//div[@class='article']//ol[@class='grid_view']/li")
        # 循环电影的条目
        for i_item in movie_list:
            # 导入item文件——>存放爬取的资料
            douban_item = DoubanItem()
            # 写详细的xpath进行数据解析
            # 如果是多行的数据，可以同split()方法处理
            douban_item['serial_number'] = i_item.xpath(
                ".//div[@class='item']//em/text()").extract_first()
            douban_item['movie_name'] = i_item.xpath(
                ".//div[@class='info']/div[@class='hd']/a/span[1]/text()"
            ).extract_first()
            content = i_item.xpath(
                ".//div[@class='info']//div[@class='bd']/p[1]/text()"
            ).extract()
            for i_content in content:
                content_s = "".join(i_content.split())
                douban_item['introduce'] = content_s
            douban_item['star'] = i_item.xpath(
                ".//span[@class='rating_num']/text()").extract_first()
            douban_item['evaluate'] = i_item.xpath(
                ".//span[@class='']/text()").extract_first()
            douban_item['describe'] = i_item.xpath(
                ".//p[@class='quote']//span/text()").extract_first()
            # 将数据yeild到pipeline
            yield douban_item
        # 解析下一页，取后页的xpath
        next_link = response.xpath(
            "//span[@class='next']/link/@href").extract()
        if next_link:
            next_link = next_link[0]
            yield scrapy.Request("https://movie.douban.com/top250" + next_link,
                                 callback=self.parse)
