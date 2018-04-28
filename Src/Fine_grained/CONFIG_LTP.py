# coding=utf-8
# encoding=utf8
from __future__ import print_function

import sys

reload(sys)
sys.setdefaultencoding('utf8')


class Config_ltp(object):
    # 基本设置
    def __init__(self):
        self.PROJECT_PATH = 'D:/coding/Projects/Fine_grained/'

        self.LTP_PATH = self.PROJECT_PATH + 'Ltp/'

        self.LTP_CWS_NAME = 'cws_cmdline'
        self.LTP_CWS_THREAD = '10'

        self.LTP_POS_NAME = 'pos_cmdline'
        self.LTP_POS_THREAD = '10'

        self.LTP_PAR_NAME = 'par_cmdline'
        self.LTP_PAR_THREAD = '10'


CONF_LTP = Config_ltp()
