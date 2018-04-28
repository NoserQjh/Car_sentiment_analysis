# coding=utf-8
# encoding=utf8
from __future__ import print_function

import sys

reload(sys)
sys.setdefaultencoding('utf8')

import re
import os
import copy

from CONFIG import CONF
from CONFIG_LTP import CONF_LTP


def ltp_commend(commend_name, num_threads, input_path, output_path):
    """LTP指令"""

    # 指令1前往目标文件夹
    command_1 = 'cd ' + CONF_LTP.LTP_PATH

    # 指令2运行ltp
    command_2 = commend_name + ' --threads ' + num_threads + ' --input ' + input_path + ' > ' + output_path
    command = command_1 + ' & ' + command_2
    os.system(command)


def grammar_analysis(text_list, term2entity, va2attributes, term2attributes,
                     sorted_unique_words, sorted_unique_words_entities,
                     sorted_unique_words_attributes, sorted_unique_words_va):
    """对输入文本进行语法分析"""

    # 保留原输入不做修改
    text_list = copy.copy(text_list)

    # entity&attribute&va替换为id tag，避免分词时被分开
    id2word = dict()
    replace_logs = []

    if len(sorted_unique_words) == 0:
        # 将实体添加入字典
        for x in term2entity.keys():
            sorted_unique_words_entities.add(x)
        # 将属性添加入字典
        for x in term2attributes.keys():
            sorted_unique_words_attributes.add(x)
        # 将形容词添加入字典
        for x in va2attributes.keys():
            name, word = x
            sorted_unique_words_va.add(word)

        sorted_unique_words.update(sorted_unique_words_entities)
        sorted_unique_words.update(sorted_unique_words_attributes)
        sorted_unique_words.update(sorted_unique_words_va)

        def _sort_uni_len(x):
            if re.match(r'不', x, flags=0):
                return len(x) + 0.5
            else:
                return len(x)

        # 按长度排序，优先匹配较长的单词
        sorted_unique_words = list(sorted_unique_words)
        sorted_unique_words.sort(key=lambda x: _sort_uni_len(x), reverse=True)

    # 首尾加入空格并替换为xxx+数字格式，防止连续在一起出现的实体无法识别
    for idx, word in enumerate(sorted_unique_words):
        for text_idx, text in enumerate(text_list):
            if word in text:
                id2word[CONF.UNIQUE + '%d' % idx] = word
                if word not in '-'.join(replace_logs):
                    text_list[text_idx] = text.replace(word,
                                                       ' ' + CONF.UNIQUE + '%d' % idx + ' ')
                    replace_logs.append('%d' % idx)

    # LTP分词

    with open(CONF.TEXT_SEG_PATH, 'w')as outfile:
        for x in text_list:
            outfile.write(x + '\n')
        outfile.close()

    ltp_commend(commend_name=CONF_LTP.LTP_CWS_NAME, num_threads=CONF_LTP.LTP_CWS_THREAD,
                input_path=CONF.TEXT_SEG_PATH, output_path=CONF.TEXT_CWS_PATH)

    words_list = []
    with open(CONF.TEXT_CWS_PATH, 'r')as infile:
        for line in infile:
            line = line.strip('\n')
            words = line.split('\t')
            words_list.append(words)

    # 将被替换的entity&lexicon词语恢复
    unique_indices = set()
    for words in words_list:
        for idx, word in enumerate(words):
            if word in id2word:
                words[idx] = id2word[word]
                unique_indices.add(idx)

    # LTP词性判断

    with open(CONF.TEXT_CWS_PATH, 'w')as outfile:
        for words in words_list:
            for word in words:
                outfile.write(word + '\t')
            outfile.write('\n')
        outfile.close()

    ltp_commend(commend_name=CONF_LTP.LTP_POS_NAME, num_threads=CONF_LTP.LTP_CWS_THREAD,
                input_path=CONF.TEXT_CWS_PATH, output_path=CONF.TEXT_POS_PATH)

    postags_list = []
    with open(CONF.TEXT_POS_PATH, 'r')as infile:
        for line in infile:
            postags = []
            line = line.strip('\n')
            words = line.split('\t')
            for word in words:
                word = word.split('_')
                postags.append(word)
            postags_list.append(postags)

    # 对words/postags的结果进行修正
    for postags in postags_list:
        for idx, postag in enumerate(postags):
            if idx in unique_indices:
                if postag[0] in sorted_unique_words_entities:
                    postag[1] = 'n'
                if postag[0] in sorted_unique_words_attributes:
                    postag[1] = 'n'
                if postag[0] in sorted_unique_words_va:
                    postag[1] = 'a'
                if idx < len(postags) - 1:
                    if postag[0] == '好' and postag[1] == 'a' and postags[idx + 1][1] == 'a':
                        postag[1] = 'd'

    # LTP词语关系判断

    with open(CONF.TEXT_POS_PATH, 'w')as outfile:
        for postags in postags_list:
            for postag in postags:
                outfile.write(postag[0] + '_' + postag[1] + '\t')
            outfile.write('\n')
        outfile.close()

    ltp_commend(commend_name=CONF_LTP.LTP_PAR_NAME, num_threads=CONF_LTP.LTP_CWS_THREAD,
                input_path=CONF.TEXT_POS_PATH, output_path=CONF.TEXT_PAR_PATH)

    arcs_list = []
    with open(CONF.TEXT_PAR_PATH, 'r')as infile:
        arcs = []
        for line in infile:
            line = line.strip('\n')
            if line == '':
                arcs_list.append(arcs)
                arcs = []
            else:
                line = line.split('\t')
                arcs.append([line[0], line[1], int(line[2]), line[3]])

    return words_list, postags_list, arcs_list
