# coding=utf-8
# encoding=utf8
from __future__ import print_function

import sys

reload(sys)
sys.setdefaultencoding('utf8')

import copy

from CONFIG import CONF
from ENEITY2SENTIMENT import entities2sentiments_group
from GRAMMAR_ANALYSIS import grammar_analysis
from INIT import init
from PRETREAT import clean_text, split_sentences
from SENTIMENT_ANALYSIS import sentiment_analysis

# 记录需要保存的关键词
sorted_unique_words = set()
sorted_unique_words_entities = set()
sorted_unique_words_attributes = set()
sorted_unique_words_va = set()


def analysis_comment(text, init_data,
                     debug=False, file=sys.stdout):
    """处理单条评论的api接口
    处理流程：
        - 预处理
        - 分句
        - 逐句：
            - 情感分析
            - aspect抽取
            - 结果后处理
        - 汇总得到整个评论的结果
    """
    print('文本内容:\n', text.decode('utf-8'), '\n')

    entities = init_data['entities']
    term2entity = init_data['term2entity']
    va2attributes = init_data['va2attributes']
    term2attributes = init_data['term2attributes']
    entity2term = init_data['entity2term']
    attributes2term = init_data['attributes2term']
    va2confidence = init_data['va2confidence']
    this_entities = copy.deepcopy(entities)

    text = clean_text(text)
    sents = split_sentences(text)

    words_list, postags_list, arcs_list = grammar_analysis(text_list=sents, term2entity=term2entity,
                                                           va2attributes=va2attributes,
                                                           term2attributes=term2attributes,
                                                           sorted_unique_words=sorted_unique_words,
                                                           sorted_unique_words_entities=sorted_unique_words_entities,
                                                           sorted_unique_words_attributes=sorted_unique_words_attributes,
                                                           sorted_unique_words_va=sorted_unique_words_va)
    this_entities = sentiment_analysis(text_list=sents, words_list=words_list, arcs_list=arcs_list,
                                       entities=this_entities,
                                       term2entity=term2entity, va2attributes=va2attributes,
                                       term2attributes=term2attributes,
                                       sorted_unique_words=sorted_unique_words,
                                       sorted_unique_words_entities=sorted_unique_words_entities,
                                       sorted_unique_words_attributes=sorted_unique_words_attributes,
                                       sorted_unique_words_va=sorted_unique_words_va,
                                       entity2term=entity2term,
                                       attributes2term=attributes2term,
                                       va2confidence=va2confidence)

    print()
    sentiments = entities2sentiments_group(this_entities)
    return sentiments


if __name__ == '__main__':

    use_nn = False

    init_data = init()

    sentiments = analysis_comment(text=CONF.TEST_COMMENT, debug=True, file=None, init_data=init_data)

    for x in sentiments:
        print(x[0], x[1], x[2], x[3], x[4])
