# coding=utf-8
# encoding=utf8
from __future__ import print_function  # python2 loading print

import os

os.environ['KERAS_BACKEND'] = 'theano'  # 换成TensorFlow backend的话，加载nn模型总报错，甚是诡异……
import sys
import imp
imp.reload(sys)
import re
import copy
import  global_var as gl
import numpy as np
from itertools import chain
from collections import Counter
from more_itertools import unique_everseen
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from pyltp import Segmentor, Postagger, Parser
# from utils.misc_utils import get_args_info

try:
    import cPickle as pickle
except ImportError:
    import pickle
test_comment = "备胎简直太差了"
_LTP_DATA_DIR = r'./ltp_data'
# _LTP_DATA_DIR = r'C:\Users\yuanz\datasets\ltp_data'
_DEFAULT_ENTITY = 'DEFAULT_ENTITY'  # 用于加载sentiment lexicon，通用情感词的默认搭配填充
# ltp对象，用于分词、POS、Parser
_segmentor = Segmentor()
_postagger = Postagger()
_parser = Parser()

# 根据EntityLink的返回结果，涉及到下列概念的entity不要
STOP_CONCEPTS = {'字词', '语言', '音乐作品', '娱乐作品', '词语'}

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


def ltp_init():
    """初始化LTP工具"""
    print('初始化LTP工具...')
    cws_model_path = os.path.join(_LTP_DATA_DIR, 'cws.model')
    print(cws_model_path)
    _segmentor.load_with_lexicon(cws_model_path, 'libs/user_dict.txt')
    pos_model_path = os.path.join(_LTP_DATA_DIR, 'pos.model')
    _postagger.load(pos_model_path)
    print(pos_model_path)
    par_model_path = os.path.join(_LTP_DATA_DIR, 'parser.model')
    _parser.load(par_model_path)
    print(par_model_path)


def ltp_release():
    """释放LTP工具"""
    _segmentor.release()
    _postagger.release()
    _parser.release()


UNIQUE = 'uni'

def init(use_nn=True):
    """初始化语料库等资源"""
    product=gl.get_value('PRODUCT','汽车')
    print('正在进行初始化设置...')
    ltp_init()
    entities, term2entity = load_enititiy(whole_part_path='./KnowledgeBase/'+product+'/whole-part.txt',
                                          entitiy_synonym_path='./KnowledgeBase/'+product+'/entity-synonym.txt')
    va2attributes, term2attributes, entities = load_attribute(
        attribute_description_path='./KnowledgeBase/'+product+'/attribute-description.txt',
        attribute_synonym_path='./KnowledgeBase/'+product+'/attribute-synonym.txt',
        entity_attribute_path='./KnowledgeBase/'+product+'/entity-attribute.txt',
        entities=entities)
    print('loading nn model')
    # model1 = load_model('libs/aspect-model.h5') if use_nn else None
    # model2 = load_model('libs/sentiment-model.h5') if use_nn else None
    # nn_kwargs1 = pickle.load(open('libs/aspect-nnargs.pkl', 'rb')) if use_nn else None
    # nn_kwargs2 = pickle.load(open('libs/sentiment-nnargs.pkl', 'rb')) if use_nn else None
    print('初始化设置成功！\n')
    return {
        'entities': entities,
        'term2entity': term2entity,
        'va2attributes': va2attributes,
        'term2attributes': term2attributes,
        #   'model1': model1,
        #   'nn_kwargs1': nn_kwargs1,
        #   'model2': model2,
        #   'nn_kwargs2': nn_kwargs2
    }


class entity(object):
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
        # 加入子级实体
        self.sons.add(new_son)

    def add_attribute(self, new_attribute):
        # 加入实体属性
        self.attributes.add(new_attribute)

    def account(self):
        # 计算各实体的评论数量

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


class attribute(object):
    # 属性类
    def __init__(self, name, father):
        # 分别对应属性名，实体，好评数，差评数
        self.name = name
        self.father = father
        self.good_num = 0
        self.bad_num = 0
        self.normal_num = 0
        self.notsure_num = 0
        self.good_comments = set()
        self.bad_comments = set()
        self.normal_comments = set()


def load_enititiy(whole_part_path, entitiy_synonym_path):
    """加载预定义的实体设置
    whole-part.txt文件格式：
        - 每行第一个词语为父级实体
        - tab之后为若干用空格隔开的子级实体
    entitiy-synonym.txt文件格式：
        - 每行第一个词语为实体名
        - tab之后为若干用空格隔开的同义词
    """
    print("加载预定义的entity设置...")
    # 加载实体间关系
    entities = []
    with open(whole_part_path, 'r',encoding='utf8') as fr:
        for line in fr:
            words = line.split('\t')
            entities = entities + [entity(name=words[0])]

    num = 0
    with open(whole_part_path, 'r',encoding='utf8') as fr:
        for line in fr:
            line = line.strip('\n')
            line = line.strip('\t')
            line = line.strip('\r')
            line = line.split('\t')
            if len(line) != 1:
                line1 = line[1]
                words = line1.split(' ')
                for new_son in words:
                    for i in range(0, len(entities) - 1):
                        if entities[i].name == new_son:
                            entities[num].add_son(entities[i])
                            entities[i].father = entities[num]
                            # print('father: ',entities[num].name,'\tson: ',entities[i].name,)
            num = num + 1

    # 加载实体与同义词关系
    term2entity = dict()
    with open(entitiy_synonym_path, 'r',encoding='utf8') as fr:
        for line in fr:
            line = line.strip().lower()
            line = line.strip('\n')
            line = line.strip('\t')
            line = line.split('\t')
            name = line[0]
            words = line[1].split(' ')
            for x in entities:
                if x.name == name:
                    for word in words:
                        term2entity[word] = x.name
                        # print('entity: ',x.name,'\tword: ',word)
    print("entity设置加载成功")
    return entities, term2entity


def load_attribute(attribute_description_path, attribute_synonym_path, entity_attribute_path, entities):
    """加载预定义的属性设置
    attribute-descrpition.txt文件格式：
        - 每行第一个词语为属性名
        - tab之后为若干用空格隔开的属性形容词，每三行描述一个属性，分别代表好，中，差
    """
    print("加载预定义的attribute设置...")
    # 加载形容词属性之间关系
    va2attributes = dict()
    with open(attribute_description_path, 'r',encoding='utf8') as fr:
        num = 0
        for line in fr:
            line = line.strip().lower()
            line = line.strip('\n')
            line = line.strip('\t')
            line = line.split('\t')
            name = line[0]
            if len(line) > 1:
                words = line[1].split(' ')
                if name == r'整体性':
                    name=name
                for word in words:
                    va2attributes[name, word] = 1 - (num % 3)
            num = num + 1

    # 加载属性与同义词关系
    term2attributes = dict()
    with open(attribute_synonym_path, 'r',encoding='utf8') as fr:
        for line in fr:
            line = line.strip().lower()
            line = line.strip('\n')
            line = line.strip('\t')
            line = line.split('\t')
            name = line[0]
            words = line[1].split(' ')
            for word in words:
                term2attributes[word] = name
                # print('attribute: ',name,'\tword: ',word)

    # 加载属性与实体间关系
    with open(entity_attribute_path, 'r',encoding='utf8') as fr:
        for line in fr:
            line = line.strip().lower()
            line = line.strip('\n')
            line = line.strip('\t')
            line = line.split('\t')
            if len(line)<2:
                continue
            name = line[0]
            words = line[1].split(' ')
            for x in entities:
                if x.name == name:
                    for word in words:
                        new_attribute = attribute(name=word, father=x)
                        x.add_attribute(new_attribute=new_attribute)
                        # print('entity: ',x.name,'\tattribute: ',word)
    print("attribute设置加载成功")
    return va2attributes, term2attributes, entities


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
    # 去掉语句中的语气前缀词语，如“非常非常非常非常好看” -> “好看”
    for word in set(PREFIX_STOPWORDS):
        if text.startswith(word):
            while text.startswith(word) and len(text) > len(word):
                text = text.replace(word, '', 1)
    # 去掉语句中的语气后缀词语
    for word in set(SUFFIX_STOPWORDS):
        if text.endswith(word):
            while text.endswith(word) and len(text) > len(word):
                text = text[::-1].replace(word[::-1], '', 1)[::-1]
    text = _domain_specific_clean(text)
    return text


def _domain_specific_clean(text):
    """领域特定的文本预处理比如众多不同表述形式的Surface，统一处理后可以提升匹配率"""
    text = re.sub(r'(new)?( )?(surface( )?(pro)?|sp)( )?[345]?( )?', r'surface', text)
    text = re.sub(r'surface( )?pen', r'surfacepen', text)
    text = re.sub(r'win(dows)?( )?10', r'windows10', text)
    return text


def split_sentences(text):
    """文本拆分为单句。后续分析时按照单句处理"""
    '''，。！!？?~～：:；;…=\s\n'''
    sents = re.split(u'[‚.，。！!？?~～：:；;…=\s\n]', text)
    sents = [sent.strip() for sent in sents if len(sent.strip()) > 0]
    return sents




sorted_unique_words = None
sorted_unique_words_entities = None
sorted_unique_words_attributes = None
sorted_unique_words_va = None


def grammar_analysis(text, entities, term2entity, va2attributes, term2attributes):
    # entity&attribute&va替换为id tag，避免分词时被分开
    id2word = dict()
    replace_logs = []
    global sorted_unique_words
    global sorted_unique_words_entities
    global sorted_unique_words_attributes
    global sorted_unique_words_va

    if not sorted_unique_words:
        sorted_unique_words = set()
        sorted_unique_words_entities = set()
        sorted_unique_words_attributes = set()
        sorted_unique_words_va = set()
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
        sorted_unique_words = list(sorted_unique_words)
        sorted_unique_words.sort(key=len, reverse=True)  # 按长度排序，优先匹配较长的单词
        for idx, word in enumerate(sorted_unique_words):
            if re.match(r'不', word, flags=0):
                sorted_unique_words.remove(word)
                sorted_unique_words.append(word)
    if 'q' in enumerate(sorted_unique_words):
        print('q in sort')
    for idx, word in enumerate(sorted_unique_words):
        if word in text:
            id2word[UNIQUE + '%d' % idx] = word
            if word not in '-'.join(replace_logs):
                text = text.replace(word, ' ' + UNIQUE + '%d' % idx + ' ')  # 首尾加入空格，防止连续在一起出现的实体无法识别
                replace_logs.append('%d' % idx)

    # print(text)
    words = list(_segmentor.segment(text.encode('utf-8')))
    # for x in words:
    #     print(x)
    # 将被替换的entity&lexicon词语恢复
    unique_indices = set()
    for idx, word in enumerate(words):
        if word in id2word:
            words[idx] = id2word[word]
            unique_indices.add(idx)

    # postags
    postags = list(_postagger.postag(words))

    # 对words/postags的结果进行修正
    for idx, (word, postag) in enumerate(zip(words, postags)):
        if idx < len(postags) - 1:
            if word == '好' and postag == 'a' and postags[idx + 1] == 'a':
                postags[idx] = 'd'
        if idx in unique_indices:
            if words[idx] in sorted_unique_words_entities:
                postags[idx] = 'n'
            if words[idx] in sorted_unique_words_attributes:
                postags[idx] = 'n'
            if words[idx] in sorted_unique_words_va:
                postags[idx] = 'v'

    # parser
    arcs = _parser.parse(words, postags)

    '''
    for i in range(0,len(arcs)):
        print(words[i], '\tPostag: ', postags[i], '\tParser: ', arcs[i].head, '\t', arcs[i].relation)
    print()
    '''
    return words, postags, arcs


def sentiment_analysis(text, words, postags, arcs,
                       entities, term2entity, va2attributes, term2attributes,result_list,
                       debug=False, file=sys.stdout):
    """情感分析模块
    算法思路：根据Dependency Parser的结果，结合一系列预定义的语法规则，抽取情感搭配
    """
    # print()
    # print(text)
    words.append('HED')
    parcs = [(arc.relation, (arc.head - 1, words[arc.head - 1]), (idx, words[idx])) for idx, arc in enumerate(arcs)]

    # for x in parcs:
    #	print(x[0],'\t',x[1][0],'\t',x[1][1],'\t',x[2][0],'\t',x[2][1])

    def _get_entity(_name):
        # 由entity名字获得相应entity
        for x in entities:
            if x.name == _name:
                return x
        return None

    def _get_attribute(_entity, _name):
        # 由entity及attribute名字获得相应attribute
        for x in _entity.attributes:
            if x.name == _name:
                return x
        return None

    def _get_score(_attribute, _va):
        # 由attibute名字及va获得相应score
        return va2attributes.setdefault((_attribute, _va), None)

    def _get_this_entity(wordnum):
        have_father = False
        fathernum = None
        fathername = None
        for parc in parcs:
            if parc[0] == 'ATT':
                if parc[1][0] == wordnum:
                    have_father = True
                    fathernum = parc[2][0]
                    fathername = parc[2][1]
                    break
        if have_father:
            if fathername in sorted_unique_words_entities:
                return fathername, fathernum
            else:
                return _get_this_entity(fathernum)
        else:
            return None, None

    # pre-process for neg and coo
    negation_logs = dict()  # 记录情感否定信息
    for parc in parcs:
        if parc[0] == 'ADV' and parc[2][1] in NEGATION_WORDS:
            negation_logs[parc[1][0]] = parc[2][1] + parc[1][1]
        if parc[0] == 'VOB' and parc[1][1] in NEGATION_WORDS:
            negation_logs[parc[2][0]] = parc[1][1] + parc[2][1]

    # sentiment pair extraction (entity -> opinion)
    got_score = False

    for parc in parcs:
        # print(parc[0],parc[1][1],parc[2][1])
        this_entity_name = None
        this_entity_num = None
        this_attribute_name = None
        this_attribute_num = None
        this_va = None
        this_va_num = None

        # 主谓/动宾/前宾
        if (parc[0] == 'VOB' or \
                        parc[0] == 'SBV' or \
                        parc[0] == 'FOB' or \
                        parc[0] == 'ADV') and \
                        parc[1][1] in sorted_unique_words_va:

            if parc[2][1] in sorted_unique_words_attributes:
                this_attribute_name = parc[2][1]
                this_attribute_num = parc[2][0]
                this_va = parc[1][1]
                this_va_num = parc[1][0]
                # 在句子里根据attribute找entity
                this_entity_name, this_entity_num = _get_this_entity(this_attribute_num)

                # if (this_entity_name is None):
                #     print('Not found entity! The attribute is ', this_attribute_name, '.\tThe va is ', this_va)
                got_score = True

            elif parc[2][1] in sorted_unique_words_entities:
                this_entity_name = parc[2][1]
                this_entity_num = parc[2][0]
                this_va = parc[1][1]
                this_va_num = parc[1][0]
                this_attribute_name = None#'整体'
                got_score = True
        # 修饰关系(定中)
        if (parc[0] == 'ATT' or \
                        parc[0] == 'CMP') and \
                        parc[2][1] in sorted_unique_words_va:
            if parc[1][1] in sorted_unique_words_attributes:
                this_attribute_name = parc[1][1]
                this_attribute_num = parc[1][0]
                this_va = parc[2][1]
                this_va_num = parc[2][0]
                # 在句子里根据attribute找entity
                this_entity_name, this_entity_num = _get_this_entity(this_attribute_num)
                # if (this_entity_name is None):
                #     print('Not found entity! The attribute is ', this_attribute_name, '.\tThe va is ', this_va)
                got_score = True

            elif parc[1][1] in sorted_unique_words_entities:
                this_entity_name = parc[1][1]
                this_entity_num = parc[1][0]
                this_va = parc[2][1]
                this_va_num = parc[2][0]
                this_attribute_name = None#'整体'
                got_score = True

        if parc[0] == 'ATT' and \
                        parc[1][1] in sorted_unique_words_va:
            if parc[2][1] in sorted_unique_words_attributes:
                this_attribute_name = parc[2][1]
                this_attribute_num = parc[2][0]
                this_va = parc[1][1]
                this_va_num = parc[1][0]
                # 在句子里根据attribute找entity
                this_entity_name, this_entity_num = _get_this_entity(this_attribute_num)
                # if this_entity_name is None:
                #     print('Not found entity! The attribute is ', this_attribute_name, '.\tThe va is ', this_va)
                got_score = True

            elif parc[2][1] in sorted_unique_words_entities:
                this_entity_name = parc[2][1]
                this_entity_num = parc[2][0]
                this_va = parc[1][1]
                this_va_num = parc[1][0]
                this_attribute_name = None
                got_score = True

        if got_score:
            # 由已获得的entity attribute va更新entity树
            "获得原本entity/attribute的name"
            if this_entity_name != None:
                this_entity_name = term2entity[this_entity_name]
            if this_attribute_name != None:
                this_attribute_name = term2attributes[this_attribute_name]

            "根据attribute推测entity"
            if this_entity_name is None:
                got_entity_name = False
                for x in entities[0].attributes:
                    if x.name == this_attribute_name:
                        this_entity_name = entities[0].name
                        this_entity = entities[0]
                        got_entity_name = True
                if got_entity_name == False:
                    unique_num = 0
                    for x in entities:
                        for y in x.attributes:
                            if y.name == this_attribute_name:
                                this_entity_name = x.name
                                unique_num = unique_num + 1
                                got_entity_name = True
                    # 仅有1entity有此attibute则将其视为当前attibute
                    '''if(unique_num > 1):
                        this_entity_name = None
                        ot_entity_name = False'''
                # if got_entity_name == True:
                #     print("got entity name:", this_entity_name)

            "根据entity及va推测attribute"
            # print(this_entity_name)
            if this_attribute_name is None:
                got_attribute_name = False
                if _get_score("整体", this_va) != None:
                    this_attribute_name = "整体"
                    got_attribute_name = True
                else:
                    unique_num = 0
                    this_entity = _get_entity(this_entity_name)
                    for x in this_entity.attributes:
                        if _get_score(x.name, this_va) != None:
                            this_attribute_name = x.name
                            unique_num = unique_num + 1
                            got_attribute_name = True
                    '''if (unique_num > 1):
                        this_attribute_name = None
                        got_attribute_name = False'''
                if got_attribute_name is True:
                    pass;#print("got attribute name:", this_attribute_name)
                else:
                    '''遍历子节点'''
                    [this_entity_name, this_attribute_name] = _get_entity(this_entity_name).if_va_father(this_va,
                                                                                                         va2attributes)
                    if this_attribute_name == None:
                        '''遍历父节点'''
                        [this_entity_name, this_attribute_name] = _get_entity(this_entity_name).if_va_son(this_va,
                                                                                                          va2attributes)
                        if this_attribute_name == None:
                            '''danger!!! 遍历填充'''
                            for x in entities:
                                for y in x.attributes:
                                    if _get_score(y.name, this_va) != None:
                                        "danger!!! changing the entity"
                                        this_entity_name = x.name
                                        this_attribute_name = y.name
                                        # unique_num = unique_num + 1
                                        got_attribute_name = True

            "根据eneity，attibute获得score"
            this_entity = _get_entity(this_entity_name)
            this_attribute = _get_attribute(this_entity, this_attribute_name)

            score = _get_score(this_attribute_name, this_va)
            "TBD 有entity or attribute 但va不匹配"
            # 否定score取反
            if score != None:
                if this_va_num in negation_logs:
                    score = score * -1
                    this_va = negation_logs.get(this_va_num)
                #score = score * (-1 if this_va_num in negation_logs else 1)
            try:
                # print('Get entity ', this_entity.name, '\tattribute ', this_attribute.name, '\tva ', this_va, '\tscore ', score)
                result_list.append([this_entity.name,this_attribute.name,this_va,score,text])
                if score == 1:
                    this_attribute.good_comments.add(this_va)
                    this_attribute.good_num = this_attribute.good_num + 1
                elif score == -1:
                    this_attribute.bad_comments.add(this_va)
                    this_attribute.bad_num = this_attribute.bad_num + 1
                elif score == 0:
                    this_attribute.normal_comments.add(this_va)
                    this_attribute.normal_num = this_attribute.normal_num + 1
                else:
                    this_attribute.notsure_num = this_attribute.notsure_num + 1
            except Exception:
                pass
        got_score = False
    words.remove('HED')  # don't forget this!
    return entities


def entities2sentiments_single(entities):
    sentiments = []
    for enti in entities:
        for atri in enti.attributes:
            if len(atri.good_comments) != 0:
                for comment in atri.good_comments:
                    sentiments.append([enti.name, atri.name, comment, 1])
            if len(atri.bad_comments) != 0:
                for comment in atri.bad_comments:
                    sentiments.append([enti.name, atri.name, comment, -1])
            if len(atri.normal_comments) != 0:
                for comment in atri.normal_comments:
                    sentiments.append([enti.name, atri.name, comment, 0])
    return sentiments


def entities2sentiments_group(entities):
    sentiments = []
    for enti in entities:
        for atri in enti.attributes:
            if len(atri.good_comments) or \
                    len(atri.bad_comments) or \
                            len(atri.normal_comments) != 0:
                sentiments.append(
                    [enti.name, atri.name, len(atri.good_comments), len(atri.bad_comments), len(atri.normal_comments)]);
    return sentiments


def analysis_comment(text,
                     debug=False, file=sys.stdout, api_debug=False, use_nn=True,**init_data):
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
    # print('文本内容：\t', text, '\n')

    entities = init_data['entities']
    term2entity = init_data['term2entity']
    va2attributes = init_data['va2attributes']
    term2attributes = init_data['term2attributes']
    text = clean_text(text)
    sents = split_sentences(text)
    # print('分句结果：')
    # for x in sents:
    #     print(x)
    # print('\n')
    sentiments = []
    result_list=[]
    this_entities = copy.deepcopy(entities)
    for sent_idx, sent in enumerate(sents):
        words, postags, arcs = grammar_analysis(text=sent, entities=this_entities, term2entity=term2entity,
                                                va2attributes=va2attributes, term2attributes=term2attributes)
        this_entities = sentiment_analysis(sent, words, postags, arcs, this_entities, term2entity, va2attributes,
                                           term2attributes, debug=debug, file=file,result_list=result_list)
    # sentiments = entities2sentiments_single(this_entities)
    sentiments = entities2sentiments_group(this_entities)
    return sentiments,result_list


if __name__ == '__main__':
    use_nn = False
    init_data = init(use_nn=use_nn)
    sentiments = analysis_comment(test_comment, debug=True, file=None, use_nn=use_nn, init_data=init_data)
    for x in sentiments:
        print(x[0], x[1], x[2], x[3], x[4])