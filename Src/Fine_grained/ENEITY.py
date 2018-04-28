# coding=utf-8
# encoding=utf8
from __future__ import print_function

import sys

reload(sys)
sys.setdefaultencoding('utf8')


class Entity(object):
    # 实体类
    def __init__(self, name):
        # 分别对应实体名，父级实体，子级实体，实体属性，好评数量，差评数量
        self.name = name
        self.father = None
        self.sons = set()
        self.attributes = set()
        self.self_good_num = 0
        self.self_bad_num = 0
        self.self_normal_num = 0
        self.self_notsure_num = 0
        self.good_num = 0
        self.bad_num = 0
        self.normal_num = 0
        self.notsure_num = 0

    def add_son(self, new_son):
        """加入子级实体"""

        self.sons.add(new_son)

    def add_attribute(self, new_attribute):
        """加入实体属性"""

        self.attributes.add(new_attribute)

    def account(self):
        """计算各实体的评论数量"""

        self.self_good_num = 0
        self.self_bad_num = 0
        self.self_normal_num = 0
        self.self_notsure_num = 0

        for x in self.attributes:
            self.self_good_num = self.self_good_num + x.good_num
            self.self_bad_num = self.self_bad_num + x.bad_num
            self.self_normal_num = self.self_normal_num + x.normal_num
            self.self_notsure_num = self.self_notsure_num + x.notsure_num

        self.good_num = self.self_good_num
        self.bad_num = self.self_bad_num
        self.normal_num = self.self_normal_num
        self.notsure_num = self.self_notsure_num

        for x in self.sons:
            x.account()
            self.good_num = self.good_num + x.good_num
            self.bad_num = self.bad_num + x.bad_num
            self.normal_num = self.normal_num + x.normal_num
            self.notsure_num = self.notsure_num + x.notsure_num

    def if_va_son(self, va, va2attributes):
        """判断子实体下是否含有属性可与va匹配"""

        for x in self.sons:
            for y in x.attributes:
                if va2attributes.setdefault((y.name, va), None) != None:
                    return [x.name, y.name]
        for x in self.sons:
            [a, b] = x.if_va_son(va, va2attributes)
            if b != None:
                return [a, b]
        return [self.name, None]

    def if_va_father(self, va, va2attributes):
        """判断父实体下是否含有属性可与va匹配"""

        if self.father != None:
            for y in self.father.attributes:

                if va2attributes.setdefault((y.name, va), None) != None:
                    return [self.father.name, y.name]
            [a, b] = self.father.if_va_father(va, va2attributes)
            if b != None:
                return [a, b]
            else:
                return [self.name, None]
        else:
            return [self.name, None]
