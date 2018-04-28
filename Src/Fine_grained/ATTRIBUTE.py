# coding=utf-8
# encoding=utf8
from __future__ import print_function
import sys
reload(sys)
sys.setdefaultencoding('utf8')


class Attribute(object):
    # 属性类
    def __init__(self, name, father):
        # 分别对应属性名，实体，好评数，差评数
        self.name = name
        self.father = father
        self.good_num = 0
        self.bad_num = 0
        self.normal_num = 0
        self.notsure_num = 0
        self.good_comments = set()
        self.bad_comments = set()
        self.normal_comments = set()
