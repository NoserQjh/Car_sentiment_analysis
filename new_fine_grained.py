# coding = utf-8
# encoding = utf8

import knowledge_base
import sys
import re
import os
import math
import numpy as np
from pyltp import Segmentor, Postagger, Parser
from preprocess import WordSet, WordEmbedding, KnowledgeBase
from sentiment_analysis import SentimentAnalysis

_LTP_DATA_DIR = r'./ltp_data_v3.4.0'
_segmentor = Segmentor()
_postagger = Postagger()
_parser = Parser()

model = SentimentAnalysis()

# 否定词前缀
NEGATION_WORDS = {'不', '无', '没', '没有', '不是', '不大', '不太'}

# 语气前缀词
PREFIX_STOPWORDS = {'感觉', '觉得', '还', '就是', '还是', '真心'}

# 语气后缀词
SUFFIX_STOPWORDS = {'了', '哈', '喔', '啊', '哈', '撒', '吧', '啦', '拉', '阿', '的', '嗷'}

# 情感强度前缀词
INTENSITY_PREFIX_WORDS = {'好', '很', '都', '真', '太', '大', '超', '挺', '还', '还挺', '特', '特别', '非常', '灰常', '都很', '相当'}

# 情感强度后缀词
INTENSITY_SUFFIX_WORDS = {'至极', '极', '透'}

# 关键词替换时前缀
UNIQUE = 'uni'

sorted_unique_words = None
sorted_unique_words_entities = None
sorted_unique_words_attributes = None
sorted_unique_words_va = None


def ltp_init():
    """初始化LTP工具"""
    print('初始化LTP工具...')
    cws_model_path = os.path.join(_LTP_DATA_DIR, 'cws.model')
    _segmentor.load_with_lexicon(cws_model_path, 'libs/user_dict.txt')
    pos_model_path = os.path.join(_LTP_DATA_DIR, 'pos.model')
    _postagger.load(pos_model_path)
    par_model_path = os.path.join(_LTP_DATA_DIR, 'parser.model')
    _parser.load(par_model_path)


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
    for word in set(PREFIX_STOPWORDS):
        if text.startswith(word):
            while text.startswith(word) and len(text) > len(word):
                text = text.replace(word, '', 1)

    # 去掉语句中的语气后缀词语
    for word in set(SUFFIX_STOPWORDS):
        if text.endswith(word):
            while text.endswith(word) and len(text) > len(word):
                text = text[::-1].replace(word[::-1], '', 1)[::-1]
    return text


def split_sentences(text):
    """文本拆分为单句。后续分析时按照单句处理"""
    '''，。！!？?~～：:；;…=\s\n'''
    sents = re.split(u'[‚.，。！!？?~～：:；;…=\s\n]', text)
    sents = [sent.strip() for sent in sents if len(sent.strip()) > 0]
    return sents


def grammar_analysis(text, knowledgebase, debug=False):
    # entity&attribute&va替换为id tag，避免分词时被分开
    id2word = dict()
    replace_logs = []
    global sorted_unique_words
    global sorted_unique_words_entities
    global sorted_unique_words_attributes
    global sorted_unique_words_descriptions

    if not sorted_unique_words:
        sorted_unique_words = set()
        sorted_unique_words_entities = set()
        sorted_unique_words_attributes = set()
        sorted_unique_words_descriptions = set()
        # 将实体添加入字典
        for x in knowledgebase.attributeSet:
            sorted_unique_words_attributes.add(x)
        # 将属性添加入字典
        for x in knowledgebase.descriptionSet:
            sorted_unique_words_descriptions.add(x)
        # 将形容词添加入字典
        for x in knowledgebase.entitySet:
            sorted_unique_words_entities.add(x)

        sorted_unique_words.update(sorted_unique_words_entities)
        sorted_unique_words.update(sorted_unique_words_attributes)
        sorted_unique_words.update(sorted_unique_words_descriptions)
        sorted_unique_words = list(sorted_unique_words)
        sorted_unique_words.sort(key=len, reverse=True)  # 按长度排序，优先匹配较长的单词
        for idx, word in enumerate(sorted_unique_words):
            if re.match(r'不', word, flags=0):
                sorted_unique_words.remove(word)
                sorted_unique_words.append(word)
    for idx, word in enumerate(sorted_unique_words):
        if word in text:
            id2word[UNIQUE + '%d' % idx] = word
            if word not in '-'.join(replace_logs):
                text = text.replace(word, ' ' + UNIQUE + '%d' % idx + ' ')  # 首尾加入空格，防止连续在一起出现的实体无法识别
                replace_logs.append('%d' % idx)

    words = list(_segmentor.segment(text.encode('utf-8')))

    sent = ''

    # 将被替换的entity&lexicon词语恢复
    unique_indices = set()
    for idx, word in enumerate(words):
        if word in id2word:
            words[idx] = id2word[word]
            unique_indices.add(idx)
        sent = sent + words[idx] + ' '

    if debug:
        print('CWS result: ' + sent)

    return sent, words


def sentiment_analysis(sent, words, knowledgebase, sentiments, debug):
    """情感分析模块
    算法思路：根据Dependency Parser的结果，结合一系列预定义的语法规则，抽取情感搭配
    """
    sas = model.analyze(sent)
    postags = None
    parcs = ()
    for sa in sas:

        target = sa[0]
        target_num = sa[2]
        description = sa[1]
        description_num = sa[3]
        polarity = sa[4]

        if debug:
            print('Get target: ' + target + ' description: ' + description)

        # target 为属性
        if (target in knowledgebase.attributeSet) | (knowledgebase.pairSA.setdefault(target, False)):
            if debug:
                print('''It's attribute''')
            if knowledgebase.pairSA.setdefault(target, False):
                target = knowledgebase.pairSA.setdefault(target, False)
            entities = knowledgebase.pairAE.setdefault(target, None)
            sentiment = list()
            in_sent = False

            for x in entities:
                confidence = 1
                if x in words:
                    in_sent = True
                    if postags is None:
                        postags = list(_postagger.postag(words))
                        arcs = _parser.parse(words, postags)
                        parcs = [(arc.relation, (arc.head - 1, words[arc.head - 1]), (idx, words[idx])) for idx, arc in
                                 enumerate(arcs)]

                    def _get_this_entity(wordnum):
                        for parc in parcs:
                            if parc[0] == 'ATT':
                                if parc[1][0] == wordnum:
                                    fathernum = parc[2][0]
                                    fathername = parc[2][1]
                                    if fathername in knowledgebase.entitySet:
                                        return fathername, fathernum
                        return None, None

                    entityname, entitynum = _get_this_entity(target_num)
                    if entityname is not None:
                        sentiment = list()
                        confidence = confidence * math.log(knowledgebase.probDA.get((description, target), 0) + 1)
                        sentiment.append({'entity': entityname,
                                          'attribute': target,
                                          'description': description,
                                          'polarity': polarity,
                                          'confidence': confidence,
                                          'description_num': description_num,
                                          'sentence': sent})
                        break
                    else:
                        in_sent = True
                        confidence = confidence * 2
                confidence = confidence * math.log(knowledgebase.probDA.get((description, target), 0) + 1)
                sentiment.append({'entity': x,
                                  'attribute': target,
                                  'description': description,
                                  'polarity': polarity,
                                  'confidence': confidence,
                                  'description_num': description_num,
                                  'sentence': sent})
            if not in_sent:
                for x in sentiment:
                    if x.get('entity') == '汽车':
                        x['confidence'] = x.get('confidence') * 2
            sentiments.append(sentiment)

        # target 为实体
        elif (target in knowledgebase.entitySet) | (knowledgebase.pairSE.setdefault(target, False)):
            if debug:
                print('''It's entity''')
            if knowledgebase.pairSE.setdefault(target, False):
                target = knowledgebase.pairSE.setdefault(target, False)
            attributes = knowledgebase.pairEA.setdefault(target, None)
            sentiment = list()
            in_sent = False

            for x in attributes:
                confidence = 1
                if x in words:
                    in_sent = True
                    if postags is None:
                        postags = list(_postagger.postag(words))
                        arcs = _parser.parse(words, postags)
                        parcs = [(arc.relation, (arc.head - 1, words[arc.head - 1]), (idx, words[idx])) for idx, arc in
                                 enumerate(arcs)]

                    def _get_this_attribute(wordnum):
                        for parc in parcs:
                            if parc[0] == 'ATT':
                                if parc[2][0] == wordnum:
                                    sonnum = parc[1][0]
                                    sonname = parc[1][1]
                                    if sonname in knowledgebase.attributeSet:
                                        return sonname, sonnum
                        return None, None

                    sonname, sonnum = _get_this_attribute(target_num)
                    if sonname is not None:
                        sentiment = list()
                        confidence = confidence * math.log(knowledgebase.probDA.get((description, sonname), 0) + 1)
                        sentiment.append({'entity': target,
                                          'attribute': sonname,
                                          'description': description,
                                          'polarity': polarity,
                                          'confidence': confidence,
                                          'description_num': description_num,
                                          'sentence': sent})
                        break
                    else:
                        in_sent = True
                        confidence = confidence * 2
                confidence = confidence * math.log(knowledgebase.probDA.get((description, x), 0) + 1)
                sentiment.append({'entity': target,
                                  'attribute': x,
                                  'description': description,
                                  'polarity': polarity,
                                  'confidence': confidence,
                                  'description_num': description_num,
                                  'sentence': sent})
            if not in_sent:
                for x in sentiment:
                    if x.get('entity') == '汽车':
                        x['confidence'] = x.get('confidence') * 2
            sentiments.append(sentiment)

        # target 不在库中
        else:
            if debug:
                print('''Wront target!''')

    # 去重
    for x in sentiments:
        for y in sentiments:
            if x is not y:
                if (len(x) == 1) & (len(y) == 1):
                    if (x[0].get('entity') == y[0].get('entity')) & (x[0].get('attribute') == y[0].get('attribute')) & \
                            (x[0].get('description_num') == y[0].get('description_num')):
                        sentiments.remove(y)

    for i in range(len(sentiments)):
        for j in range(i, len(sentiments)):
            if sentiments[i][0].get('description_num') > sentiments[j][0].get('description_num'):
                temp = sentiments[j]
                sentiments[j] = sentiments[i]
                sentiments[i] = temp
    return sentiments


def sort_sentiments(sentiments):
    this_phases = list()
    this_score = list()
    for sentiment in sentiments:
        last_score = this_score
        last_phases = this_phases
        if len(last_phases) == 0:
            for condition in sentiment:
                this_phases.append([condition])
                this_score.append(condition.get('confidence'))
        else:
            this_phases = [None for i in range(len(sentiment))]
            this_score = [0 for i in range(len(sentiment))]
            for idx1, condition in enumerate(sentiment):
                for idx2, last_condition in enumerate(last_phases):
                    if condition.get('confidence') * last_score[idx2] > this_score[idx1]:
                        this_phases[idx1] = last_phases[idx2] + [condition]
                        this_score[idx1] = condition.get('confidence') * last_score[idx2]
    result = this_phases[np.argmax(this_score)]
    return result


def analysis_comment(text, knowledgebase=None, debug=False, file=sys.stdout):
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

    text = clean_text(text)
    sents = split_sentences(text)

    sentiments = list()

    for sent_idx, sent in enumerate(sents):
        sent, words = grammar_analysis(text=sent, knowledgebase=knowledgebase, debug=debug)
        sentiments = sentiment_analysis(sent=sent, words=words, knowledgebase=knowledgebase, sentiments=sentiments,
                                        debug=debug)
    sentiments = sort_sentiments(sentiments)
    return sentiments


if __name__ == '__main__':
    ltp_init()
    knowledgebase = knowledge_base.knowledge_base_init()
    sentence = '充沛的动力、炫酷的踏板造型再加上本田雷达家族运动基因的传承'
    result = analysis_comment(sentence, knowledgebase=knowledgebase, debug=True, file=sys.stdout)
    for x in result:
        print(x)
