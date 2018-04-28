# coding=utf-8
# encoding=utf8
from __future__ import print_function
import sys

reload(sys)
sys.setdefaultencoding('utf8')


def load_words_set(path):
    result = set()
    with open(path, 'r') as fr:
        for line in fr:
            line = line.strip('\n')
            line = line.strip('\t')
            line = line.strip('\r')
            line = line.split('\t')
            for x in line:
                result.add(x)
    return result


def load_words_dict(path):
    result = dict()
    with open(path, 'r') as fr:
        for line in fr:
            line = line.strip('\n')
            line = line.strip('\t')
            line = line.strip('\r')
            line = line.split('\t')
            if len(line) > 1:
                for idx in range(len(line) - 1):
                    result[line[idx + 1]] = line[0]
    return result
