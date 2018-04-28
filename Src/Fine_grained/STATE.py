# encoding=utf8
from __future__ import print_function

import sys

reload(sys)
sys.setdefaultencoding('utf8')

import math
from CONFIG import CONF


class State(object):
    # 状态类记录评价可能蕴含情感
    def __init__(self, this_entity_name=None, this_attribute_name=None, this_va=None, this_score=0,
                 confidence=CONF.CONFIDENCE_LIMITH, text=None, unique_num=1, is_all=True):
        self.this_entity_name = this_entity_name
        self.this_attribute_name = this_attribute_name
        self.this_va = this_va
        self.this_score = this_score
        self.confidence = confidence
        self.update_confidence_unique(unique_num=unique_num, is_all=is_all)
        self.text = text

    def update_confidence_unique(self, unique_num, is_all):
        """根据unique更新confidence"""

        if unique_num != 1:
            if is_all:
                self.confidence = self.confidence / math.sqrt(unique_num)
            else:
                self.confidence = self.confidence / math.sqrt(unique_num) / 2

    def update_confidence_va(self, va2confidence):
        """根据va及attribute更新confidence"""

        self.confidence = self.confidence * va2confidence.setdefault((self.this_attribute_name, self.this_va), 0)
