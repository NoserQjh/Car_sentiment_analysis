# coding=utf-8
# encoding=utf8
from __future__ import print_function

import sys

reload(sys)
sys.setdefaultencoding('utf8')

import re
import os
import glob

from CONFIG import CONF
from CONFIG_LTP import CONF_LTP


def sentence_seg(intput_path, output_path):
    """分句"""

    with open(intput_path, 'r') as infile:
        with open(output_path, 'w') as outfile:
            for line in infile:

                line = line.strip()

                # 去除括号
                rc = re.compile('\(.*\)|\[.*\]|\{.*\}|（.*）|【.*】')
                line = rc.sub('', line)

                # 分开句子
                lsp = re.split('。|！|？|；', line)
                while '' in lsp:
                    lsp.remove('')

                # save
                for snt in lsp:
                    snt = snt.strip(u'，').strip(u'：').strip(u'~') + u'。'
                    if len(snt) > 5 and len(snt) < 200:
                        outfile.write((snt + u'\n'))
            outfile.close()
        infile.close()


def ltp_commend(commend_name, num_threads, input_path, output_path):
    """LTP指令"""

    # 指令1前往目标文件夹
    command_1 = 'cd ' + CONF_LTP.LTP_PATH

    # 指令2运行ltp
    command_2 = commend_name + ' --threads ' + num_threads + ' --input ' + input_path + ' > ' + output_path
    command = command_1 + ' & ' + command_2
    os.system(command)


def get_pairs_na(input_path, output_path):
    """根据词性判别结果找到并统计形容词属性对"""

    pair_num = dict()
    with open(input_path) as infile:
        for line in infile:
            # 按词语分开
            line = line.strip('\n')
            wps = line.split('\t')

            # 逐个词语判断是否构成pair
            for i in range(len(wps)):
                word0 = wps[i].split('_')[0].replace('.', '')
                pos0 = wps[i].split('_')[1]

                # 连续2词
                if i + 1 < len(wps):
                    word1 = wps[i + 1].split('_')[0].replace('.', '')
                    pos1 = wps[i + 1].split('_')[1]
                    if pos0 == 'n' and pos1 == 'a':
                        na = word0 + ' ' + word1
                        pair_num[na] = pair_num.setdefault(na, 0) + 1

                    # 连续3词
                    if i + 2 < len(wps):
                        word2 = wps[i + 2].split('_')[0].replace('.', '')
                        pos2 = wps[i + 2].split('_')[1]
                        if pos0 == 'n' and pos1 == 'd' and pos2 == 'a':
                            na = word0 + ' ' + word2
                            pair_num[na] = pair_num.setdefault(na, 0) + 1
        infile.close()

    # 按出现次数排序
    pn_sort = sorted(pair_num.items(), key=lambda item: (item[1], item[0]), reverse=True)

    # 输出
    with open(output_path, 'w')as outfile:
        for pn in pn_sort:
            outfile.write('%s\t%d\n' % (pn[0], pn[1]))
        outfile.close()


def get_pairs_nn(input_path, output_path):
    """根据词性判别结果找到并统计形容词属性对"""
    pair_num = dict()
    with open(input_path) as infile:
        for line in infile:
            # 按词语分开
            line = line.strip('\n')
            wps = line.split('\t')

            # 逐个词语判断是否构成pair
            for i in range(len(wps)):
                word0 = wps[i].split('_')[0].replace('.', '')
                pos0 = wps[i].split('_')[1]
                if ((word0 == '用户') or (word0 == '优点')):
                    continue
                # 连续2词
                if i + 1 < len(wps):
                    word1 = wps[i + 1].split('_')[0].replace('.', '')
                    pos1 = wps[i + 1].split('_')[1]
                    if pos0 == 'n' and pos1 == 'n':
                        na = word0 + ' ' + word1
                        pair_num[na] = pair_num.setdefault(na, 0) + 1

                    # 连续3词
                    if i + 2 < len(wps):
                        word2 = wps[i + 2].split('_')[0].replace('.', '')
                        pos2 = wps[i + 2].split('_')[1]
                        if pos0 == 'n' and pos1 == 'u' and pos2 == 'n':
                            na = word0 + ' ' + word2
                            pair_num[na] = pair_num.setdefault(na, 0) + 1
        infile.close()

    # 按出现次数排序
    pn_sort = sorted(pair_num.items(), key=lambda item: (item[1], item[0]), reverse=True)

    # 输出
    with open(output_path, 'w')as outfile:
        for pn in pn_sort:
            outfile.write('%s\t%d\n' % (pn[0], pn[1]))
        outfile.close()


def genreate_sentiment_dict(negative_path, positive_path, supplement_path):
    """生成情感词典"""

    sentiment_dict = dict()

    # 读取消极词典
    negative_dict_paths = glob.glob(os.path.join(negative_path, "*"))
    for negative_dict in negative_dict_paths:
        with open(negative_dict, 'r') as infile:
            for line in infile:
                line = line.strip('\n')
                line = line.strip('\s')
                line = line.strip(' ')
                sentiment_dict[line] = -1
            infile.close()

    # 读取积极词典
    positive_dict_paths = glob.glob(os.path.join(positive_path, "*"))
    for positive_dict in positive_dict_paths:
        with open(positive_dict, 'r') as infile:
            for line in infile:
                line = line.strip('\n')
                line = line.strip('\s')
                line = line.strip(' ')
                sentiment_dict[line] = 1
            infile.close()

    # 读取手动修正词典
    with open(supplement_path, 'r') as infile:
        for line in infile:
            line = line.strip()
            line = line.strip('\n')
            line = line.split('\t')
            word = line[0]
            score = int(line[1])
            sentiment_dict[word] = score
        infile.close()

    return sentiment_dict


def analysis_sentiment_pairs_na(sentiment_dict, input_path, output_path):
    """根据情感词典对找到的形容词属性关系进行判别"""

    with open(input_path, 'r') as infile:
        with open(output_path, 'w') as outfile:
            for line in infile:
                line = line.strip('\n')
                line = line.split('\t')
                nword = line[0].split(' ')[0]
                aword = line[0].split(' ')[1]
                num = line[1]
                score = sentiment_dict.setdefault(aword, 0)
                outfile.write(nword + '\t' + str(score) + '\t' + aword + '\t' + num + '\n')
        infile.close()


def analysis_grammar_data():
    """对数据进行语法分析"""

    # 分开句子
    sentence_seg(intput_path=CONF.REVIEWS_PATH, output_path=CONF.REVIEWS_SEG_PATH)

    # LTP分词
    ltp_commend(commend_name=CONF_LTP.LTP_CWS_NAME, num_threads=CONF_LTP.LTP_CWS_THREAD,
                input_path=CONF.REVIEWS_SEG_PATH, output_path=CONF.REVIEWS_CWS_PATH)

    # LTP词性判断
    ltp_commend(commend_name=CONF_LTP.LTP_POS_NAME, num_threads=CONF_LTP.LTP_POS_THREAD,
                input_path=CONF.REVIEWS_CWS_PATH, output_path=CONF.REVIEWS_POS_PATH)

    # LTP词语关系
    ltp_commend(commend_name=CONF_LTP.LTP_PAR_NAME, num_threads=CONF_LTP.LTP_PAR_THREAD,
                input_path=CONF.REVIEWS_POS_PATH, output_path=CONF.REVIEWS_PAR_PATH)


if __name__ == '__main__':
    # 对数据进行语法分析
    # analysis_grammar_data()

    # 寻找名词形容词词对
    # get_pairs_na(input_path=CONF.REVIEWS_POS_PATH, output_path=CONF.PAIRS_NA_PATH)

    # 寻找名词名词词对
    # get_pairs_nn(input_path=CONF.REVIEWS_POS_PATH, output_path=CONF.PAIRS_NN_PATH)

    # 读取情感词典
    sentiment_dict = genreate_sentiment_dict(negative_path=CONF.NEGATIVE_DICT_PATH,
                                             positive_path=CONF.POSITIVE_DICT_PATH,
                                             supplement_path=CONF.SUPPLEMENT_DICT_PATH)

    # 根据情感词典对形容词对做标注
    analysis_sentiment_pairs_na(sentiment_dict=sentiment_dict,
                                input_path=CONF.PAIRS_NA_PATH, output_path=CONF.NA_OUT_PATH)
