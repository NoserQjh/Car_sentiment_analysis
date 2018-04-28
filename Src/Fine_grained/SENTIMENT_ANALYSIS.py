# coding=utf-8
# encoding=utf8
from __future__ import print_function

import sys

reload(sys)
sys.setdefaultencoding('utf8')
import re
import STATE
import copy
import gensim
import numpy as np
from CONFIG import CONF

# 否定词前缀
NEGATION_WORDS = {'不', '无', '没', '没有', '不是', '不大', '不太'}


class Phase:
    def __init__(self, states_list, confidence):
        self.states_list = states_list
        self.confidence = confidence


def sentiment_analysis(text_list, words_list, arcs_list, entities, term2entity, va2attributes, term2attributes,
                       sorted_unique_words, sorted_unique_words_entities,
                       sorted_unique_words_attributes, sorted_unique_words_va,
                       entity2term, attributes2term, va2confidence):
    """情感分析模块
    算法思路：根据Dependency Parser的结果，结合一系列预定义的语法规则，抽取情感搭配
    """

    def _get_entity(_name):
        """由entity名字获得相应entity"""
        for x in entities:
            if x.name == _name:
                return x
        return None

    def _get_attribute(_entity, _name):
        """由entity及attribute名字获得相应attribute"""
        for x in _entity.attributes:
            if x.name == _name:
                return x
        return None

    def _get_score(_attribute, _va):
        """由attibute名字及va获得相应score"""
        return va2attributes.setdefault((_attribute, _va), None)

    def _get_this_entity(wordnum):
        """根据name获得entity"""
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

    def _get_this_attribute(wordnum):
        """根据name获得attribute"""
        have_son = False
        sonnum = None
        sonname = None
        for parc in parcs:
            if parc[0] == 'ATT':
                if parc[2][0] == wordnum:
                    have_son = True
                    sonnum = parc[1][0]
                    sonname = parc[1][1]
                    break
        if have_son:
            if sonname in sorted_unique_words_attributes:
                return sonname, sonnum
            else:
                return _get_this_attribute(sonnum)
        else:
            return None, None

    def _analysis_parc(parc):
        """分析parc是否含有评价信息"""
        got_score = False
        this_entity_name = None
        this_attribute_name = None
        this_va = None
        this_va_num = None
        # 主谓/动宾/前宾
        if ((parc[0] == 'VOB' or
                     parc[0] == 'SBV' or
                     parc[0] == 'FOB' or
                     parc[0] == 'ADV') and
                    parc[1][1] in sorted_unique_words_va):

            if parc[2][1] in sorted_unique_words_attributes:
                this_attribute_name = parc[2][1]
                this_attribute_num = parc[2][0]
                this_va = parc[1][1]
                this_va_num = parc[1][0]

                # 在句子里根据attribute找entity
                this_entity_name, this_entity_num = _get_this_entity(this_attribute_num)
                if (this_entity_name == None):
                    print('Not found entity! The attribute is ', this_attribute_name, '.\tThe va is ', this_va)
                got_score = True

            elif parc[2][1] in sorted_unique_words_entities:
                this_entity_name = parc[2][1]
                this_entity_num = parc[2][0]
                this_va = parc[1][1]
                this_va_num = parc[1][0]

                # 在句子里根据entity找attribute
                this_attribute_name, this_attribute_num = _get_this_entity(this_entity_num)
                if (this_entity_name == None):
                    print('Not found entity! The attribute is ', this_attribute_name, '.\tThe va is ', this_va)
                got_score = True

        # 修饰关系(定中)
        if ((parc[0] == 'ATT' or
                     parc[0] == 'CMP') and
                    parc[2][1] in sorted_unique_words_va):
            if parc[1][1] in sorted_unique_words_attributes:
                this_attribute_name = parc[1][1]
                this_attribute_num = parc[1][0]
                this_va = parc[2][1]
                this_va_num = parc[2][0]

                # 在句子里根据attribute找entity
                this_entity_name, this_entity_num = _get_this_entity(this_attribute_num)
                if (this_entity_name == None):
                    print('Not found entity! The attribute is ', this_attribute_name, '.\tThe va is ', this_va)
                got_score = True

            elif parc[1][1] in sorted_unique_words_entities:
                this_entity_name = parc[1][1]
                this_entity_num = parc[1][0]
                this_va = parc[2][1]
                this_va_num = parc[2][0]

                # 在句子里根据entity找attribute
                this_attribute_name, this_attribute_num = _get_this_entity(this_entity_num)
                if (this_entity_name == None):
                    print('Not found entity! The attribute is ', this_attribute_name, '.\tThe va is ', this_va)
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
                if (this_entity_name == None):
                    print('Not found entity! The attribute is ', this_attribute_name, '.\tThe va is ', this_va)
                got_score = True

            elif parc[2][1] in sorted_unique_words_entities:
                this_entity_name = parc[2][1]
                this_entity_num = parc[2][0]
                this_va = parc[1][1]
                this_va_num = parc[1][0]

                # 在句子里根据entity找attribute
                this_attribute_name, this_attribute_num = _get_this_entity(this_entity_num)
                if (this_entity_name == None):
                    print('Not found entity! The attribute is ', this_attribute_name, '.\tThe va is ', this_va)
                got_score = True

        return got_score, this_entity_name, this_attribute_name, this_va, this_va_num

    def _guess_entity(this_attribute_name, this_va, confidence):
        states = set()
        unique_num = 0
        temp_this_entity_name = list()

        for x in entities:
            for y in x.attributes:
                if y.name == this_attribute_name:
                    temp_this_entity_name.append(x.name)
                    unique_num = unique_num + 1

        # 仅有1entity有此attibute则将其视为当前entity
        if unique_num == 1:
            states.add(
                STATE.State(this_entity_name=temp_this_entity_name[0],
                            this_attribute_name=this_attribute_name,
                            this_va=this_va,
                            confidence=confidence,
                            text=text))
        else:
            for x in temp_this_entity_name:
                if x == entities[0].name:
                    states.add(STATE.State(this_entity_name=x,
                                           this_attribute_name=this_attribute_name,
                                           this_va=this_va,
                                           confidence=confidence,
                                           text=text,
                                           unique_num=unique_num,
                                           is_all=True))
                else:
                    states.add(STATE.State(this_entity_name=x,
                                           this_attribute_name=this_attribute_name,
                                           this_va=this_va,
                                           confidence=confidence,
                                           text=text,
                                           unique_num=unique_num,
                                           is_all=False))
        return states

    def _guess_attribute(this_entity_name, this_va, confidence, va2attributes):
        states = set()
        got_attribute_name = False
        unique_num = 0
        temp_this_attribute_name = list()

        this_entity = _get_entity(this_entity_name)

        for x in this_entity.attributes:
            if _get_score(x.name, this_va) != None:
                temp_this_attribute_name.append(x.name)
                unique_num = unique_num + 1
                got_attribute_name = True

        # 当前entity下仅有1attribute对应va则将其视为当前attribute
        if unique_num == 1:
            states.add(
                STATE.State(this_entity_name=this_entity_name,
                            this_attribute_name=temp_this_attribute_name[0],
                            this_va=this_va,
                            confidence=confidence,
                            text=text))
        else:
            for x in temp_this_attribute_name:
                if x == "整体":
                    states.add(STATE.State(this_entity_name=this_entity_name,
                                           this_attribute_name=x,
                                           this_va=this_va,
                                           confidence=confidence,
                                           text=text,
                                           unique_num=unique_num,
                                           is_all=True))
                else:
                    states.add(STATE.State(this_entity_name=this_entity_name,
                                           this_attribute_name=x,
                                           this_va=this_va,
                                           confidence=confidence,
                                           text=text,
                                           unique_num=unique_num,
                                           is_all=False))

        if got_attribute_name == False:
            'confidence'

            '''遍历子节点'''
            [this_entity_name, temp_this_attribute_name] = _get_entity(this_entity_name).if_va_father(this_va,
                                                                                                      va2attributes)
            if temp_this_attribute_name:
                states.add(
                    STATE.State(this_entity_name=this_entity_name,
                                this_attribute_name=temp_this_attribute_name,
                                this_va=this_va,
                                confidence=0,
                                text=text))
            else:
                '''遍历父节点'''
                [this_entity_name, temp_this_attribute_name] = _get_entity(this_entity_name).if_va_son(this_va,
                                                                                                       va2attributes)
                if temp_this_attribute_name:
                    states.add(
                        STATE.State(this_entity_name=this_entity_name,
                                    this_attribute_name=temp_this_attribute_name,
                                    this_va=this_va,
                                    confidence=0,
                                    text=text))
                else:
                    '''danger!!! 遍历填充'''
                    for x in entities:
                        for y in x.attributes:
                            if _get_score(y.name, this_va) != None:
                                print("danger!!! changing the entity")
                                states.add(
                                    STATE.State(this_entity_name=x.name,
                                                this_attribute_name=y.name,
                                                this_va=this_va,
                                                confidence=0,
                                                text=text))
        return states

    phases = list()

    for text_idx in range(len(words_list)):
        text = text_list[text_idx]
        words = words_list[text_idx]
        arcs = arcs_list[text_idx]

        # 对arcs进行预处理
        arcs.insert(0, ['HBV', 'HBV', 0, 'HBV'])
        parcs = []
        for idx, arc in enumerate(arcs):
            new_arc = [arc[3], [arc[2], arcs[arc[2]][0]], [idx, arc[0]]]
            parcs.append(new_arc)

        # 记录情感否定信息
        negation_logs = dict()
        for parc in parcs:
            if parc[0] == 'ADV' and parc[2][1] in NEGATION_WORDS:
                negation_logs[parc[1][0]] = parc[2][1] + parc[1][1]
            if parc[0] == 'VOB' and parc[1][1] in NEGATION_WORDS:
                negation_logs[parc[2][0]] = parc[1][1] + parc[2][1]

        for parc in parcs:

            got_score, this_entity_name, this_attribute_name, this_va, this_va_num = _analysis_parc(parc=parc)

            if got_score:
                # 由已获得的entity attribute va更新entity树

                states = set()

                # 置信度
                confidence = 1.

                # 获得原本entity/attribute的name
                if this_entity_name != None:
                    this_entity_name = term2entity[this_entity_name]
                else:
                    confidence = confidence / 2

                if this_attribute_name != None:
                    this_attribute_name = term2attributes[this_attribute_name]
                else:
                    confidence = confidence / 2
                if (this_entity_name != None) & (this_attribute_name != None):
                    if _get_attribute(_get_entity(this_entity_name), this_attribute_name):
                        states.add(STATE.State(this_entity_name=this_entity_name,
                                               this_attribute_name=this_attribute_name,
                                               this_va=this_va,
                                               confidence=confidence))
                    else:
                        states.update(
                            _guess_entity(this_attribute_name=this_attribute_name, this_va=this_va,
                                          confidence=confidence))
                        states.update(
                            _guess_attribute(this_entity_name=this_entity_name, this_va=this_va, confidence=confidence,
                                             va2attributes=va2attributes))

                # 根据attribute推测entity
                if this_entity_name == None:
                    states.update(
                        _guess_entity(this_attribute_name=this_attribute_name, this_va=this_va, confidence=confidence))

                # 根据entity及va推测attribute
                if this_attribute_name == None:
                    states.update(
                        _guess_attribute(this_entity_name=this_entity_name, this_va=this_va, confidence=confidence,
                                         va2attributes=va2attributes))

                # va attribute 不匹配怎么办？

                # states confidence更新
                for state in states:
                    for word in entity2term.setdefault(state.this_entity_name, []):
                        pattern = unicode(word, 'utf-8')
                        if re.search(pattern=pattern, string=text):
                            state.confidence = state.confidence * 3

                    for word in attributes2term.setdefault(state.this_attribute_name, []):
                        pattern = unicode(word, 'utf-8')
                        if re.search(pattern=pattern, string=text):
                            state.confidence = state.confidence * 3

                for state in states:
                    state.update_confidence_va(va2confidence)
                    # 判断极性
                    state.this_score = _get_score(this_attribute_name, this_va)
                    # 否定score取反
                    if state.this_score:
                        state.this_score = state.this_score * (-1 if this_va_num in negation_logs else 1)

                states.add(STATE.State())

                # states根据confidence排序
                states = list(states)
                states.sort(key=lambda x: x.confidence, reverse=True)

                if len(states) > 1:
                    phases.append(states)

                ''''# 获取当前最可能的state信息
                this_entity_name = states[0].this_entity_name
                this_attribute_name = states[0].this_attribute_name
                confidence = states[0].confidence

                # 根据eneity，attibute获得score
                this_entity = _get_entity(this_entity_name)
                this_attribute = _get_attribute(this_entity, this_attribute_name)
                score = _get_score(this_attribute_name, this_va)

                # 否定score取反
                if score != None:
                    score = score * (-1 if this_va_num in negation_logs else 1)

                # confidence足够则更新entity树
                if confidence > CONF.CONFIDENCE_LIMITH:
                    print('Get entity ', this_entity.name, '\tattribute ', this_attribute.name, '\tva ', this_va,
                          '\tscore ', score, '\tconfidence %.5f' % confidence)

                    # 更新entity树
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
                else:
                    print('Get entity ', this_entity.name, '\tattribute ', this_attribute.name, '\tva ', this_va,
                          '\tscore ', score, '\tconfidence %.5f\tconfidence not enough!' % confidence)'''

    model = gensim.models.Word2Vec.load(CONF.WORD2VEC_PATH)

    phase = list()
    for idx, this_phase in enumerate(phases):
        last_phase = phase
        phase = list()
        if idx == 0:
            for state in this_phase:
                phase.append(Phase([state], state.confidence))
        else:
            for this_state in this_phase:
                new_phase = Phase([], 0)
                for last_state in last_phase:

                    entity_confidence = 0.5
                    if (last_state.states_list[-1].this_entity_name in model.wv.index2entity) & (
                                this_state.this_entity_name in model.wv.index2entity):
                        entity_confidence = entity_confidence + model.similarity(
                            unicode(last_state.states_list[-1].this_entity_name, "utf-8"),
                            unicode(this_state.this_entity_name, "utf-8"))

                    attribute_confidence = 0.5
                    if (last_state.states_list[-1].this_attribute_name in model.wv.index2entity) & (
                                this_state.this_attribute_name in model.wv.index2entity):
                        attribute_confidence = attribute_confidence + model.similarity(
                            unicode(last_state.states_list[-1].this_attribute_name, "utf-8"),
                            unicode(this_state.this_attribute_name, "utf-8"))

                    new_confidence = np.log(last_state.confidence + 1e-5) + np.log(this_state.confidence + 1e-5) + \
                                     (np.log(entity_confidence + 1e-5) + np.log(attribute_confidence + 1e-5))*10
                    if new_confidence > new_phase.confidence:
                        new_phase.confidence = new_confidence
                        new_phase.states_list = copy.copy(last_state.states_list)
                        new_phase.states_list.append(this_state)
                phase.append(new_phase)

    phase.sort(key=lambda x: x.confidence, reverse=True)
    phase = phase[0]

    print('\nresult:\n')

    for y in phase.states_list:
        print(y.this_entity_name, y.this_attribute_name, y.this_va, y.confidence, y.text)
    return entities
