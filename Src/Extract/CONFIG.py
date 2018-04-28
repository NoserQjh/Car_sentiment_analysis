# coding=utf-8
# encoding=utf8
from __future__ import print_function

import sys

reload(sys)
sys.setdefaultencoding('utf8')


class Config(object):
    # 基本设置
    def __init__(self):
        # porject path
        self.PROJECT_PATH = 'D:/coding/Projects/Fine_grained/'

        # extract path
        self.EXTRACT_PATH = self.PROJECT_PATH + 'Src/Extract/'

        # data path
        self.EXTRACT_DATA_PATH = self.EXTRACT_PATH + 'Extract_data/'
        self.REVIEWS_PATH = self.EXTRACT_DATA_PATH + 'reviews.txt'

        # temp path
        self.TEMP_PATH = self.EXTRACT_PATH + 'Temp/'
        self.REVIEWS_SEG_PATH = self.TEMP_PATH + 'reviews_seg.txt'
        self.REVIEWS_CWS_PATH = self.TEMP_PATH + 'reviews_cws.txt'
        self.REVIEWS_POS_PATH = self.TEMP_PATH + 'reviews_pos.txt'
        self.REVIEWS_PAR_PATH = self.TEMP_PATH + 'reviews_par.txt'
        self.PAIRS_NA_PATH = self.TEMP_PATH + 'pairs_na.txt'
        self.PAIRS_NN_PATH = self.TEMP_PATH + 'pairs_nn.txt'

        # out path
        self.EXTRACT_OUT_PATH = self.EXTRACT_PATH + 'Extract_out/'
        self.NA_OUT_PATH = self.EXTRACT_OUT_PATH + 'na_out.txt'
        self.NN_OUT_PATH = self.EXTRACT_OUT_PATH + 'nn_out.txt'
        # self.LIB_PATH = self.PROJECT_PATH + 'Libs/'
        # self.NA_OUT_PATH = self.LIB_PATH + 'na_out.txt'

        # dict path
        self.EXTRACT_SENTIMENT_PATH = self.EXTRACT_PATH + 'Extract_sentiment/'
        self.NEGATIVE_DICT_PATH = self.EXTRACT_SENTIMENT_PATH + 'Negative'
        self.POSITIVE_DICT_PATH = self.EXTRACT_SENTIMENT_PATH + 'Positive'
        self.SUPPLEMENT_DICT_PATH = self.EXTRACT_SENTIMENT_PATH + 'supplement.txt'

        # word2vec path
        self.WORD2VEC_PATH = self.PROJECT_PATH + 'Word2vec/word2vec_wx'
        self.WORD2VEC_PATH_NEW = self.PROJECT_PATH + 'Word2vec/word2vec'
        self.WORD2VEC_PATH_ADD = self.PROJECT_PATH + 'Word2vec/word2vec_add'

CONF = Config()
