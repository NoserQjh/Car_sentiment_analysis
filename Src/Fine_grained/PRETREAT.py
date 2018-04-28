# coding=utf-8
# encoding=utf8
from __future__ import print_function
import sys

reload(sys)
sys.setdefaultencoding('utf8')

import re
from CONFIG import CONF
from UTILS import load_words_set, load_words_dict


def clean_text(text):
    """文本去噪"""

    text = text.lower()
    text = text.replace(r'\n', ' ')
    text = text.replace(r'&hellip;', ' ')
    text = re.sub(r'\.{2,}', '，', text)  # many dots to comma
    text = re.sub(r'[1-9二三四五六七八九]、', ' ', text)
    text = re.sub(r' +', r' ', text)  # many spaces to one
    text = re.sub(r'不(是很|太)', r'不', text)
    text = re.sub(r'没(有)?想象(中)?(的)?', r'不', text)
    text = re.sub(r'简直', r'', text)

    # 去掉语句中的语气前缀词语
    PREFIX_WORDS = load_words_set(CONF.PREFIX_PATH)
    for word in set(PREFIX_WORDS):
        if text.startswith(word):
            while text.startswith(word) and len(text) > len(word):
                text = text.replace(word, '', 1)

    # 去掉语句中的语气后缀词语
    SUFFIX_WORDS = load_words_set(CONF.SUFIX_PATH)
    for word in set(SUFFIX_WORDS):
        if text.endswith(word):
            while text.endswith(word) and len(text) > len(word):
                text = text[::-1].replace(word[::-1], '', 1)[::-1]

    SUB_WORDS = load_words_dict(CONF.SUB_PATH)
    for subword in SUB_WORDS:
        text = re.sub(subword, SUB_WORDS.setdefault(subword, ''), text)

    return text


def split_sentences(text):
    """文本拆分为单句。后续分析时按照单句处理"""

    '''，。！!？?~～：:；;…=\s\n'''
    sents = re.split(u'[‚.，。！!？?~～：:；;…=\s\n]', text.decode('utf-8'))
    sents = [sent.strip() for sent in sents if len(sent.strip()) > 0]
    return sents
