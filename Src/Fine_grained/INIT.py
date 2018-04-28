# coding=utf-8
# encoding=utf8
from __future__ import print_function

import sys

reload(sys)
sys.setdefaultencoding('utf8')

import ATTRIBUTE
import ENEITY
from CONFIG import CONF


def load_enititiy(whole_part_path, entitiy_synonym_path):
    """加载预定义的实体设置"""

    print("加载预定义的entity设置...")

    # 加载实体间关系
    entities = []
    with open(whole_part_path, 'r') as fr:
        for line in fr:
            words = line.split('\t')
            entities = entities + [ENEITY.Entity(name=words[0])]

    num = 0
    with open(whole_part_path, 'r') as fr:
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
    entity2term = dict()
    with open(entitiy_synonym_path, 'r') as fr:
        for line in fr:
            line = line.strip().lower()
            line = line.strip('\n')
            line = line.strip('\t')
            line = line.split('\t')
            name = line[0]
            words = line[1].split(' ')
            for x in entities:
                if x.name == name:
                    entity2term[name] = words
                    for word in words:
                        term2entity[word] = x.name
                        # print('entity: ',x.name,'\tword: ',word)
    print("entity设置加载成功\n")
    return entities, term2entity, entity2term


def load_attribute(attribute_synonym_path, entity_attribute_path, entities):
    """加载预定义的属性设置"""

    print("加载预定义的attribute设置...")

    # 加载属性与同义词关系
    term2attributes = dict()
    attributes2term = dict()
    with open(attribute_synonym_path, 'r') as fr:
        for line in fr:
            line = line.strip().lower()
            line = line.strip('\n')
            line = line.strip('\t')
            line = line.split('\t')
            name = line[0]
            words = line[1].split(' ')
            attributes2term[name] = words
            for word in words:
                term2attributes[word] = name
                # print('attribute: ',name,'\tword: ',word)

    # 加载属性与实体间关系
    with open(entity_attribute_path, 'r') as fr:
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
                        new_attribute = ATTRIBUTE.Attribute(name=word, father=x)
                        x.add_attribute(new_attribute=new_attribute)
                        # print('entity: ',x.name,'\tattribute: ',word)
    print("attribute设置加载成功\n")

    return term2attributes, attributes2term, entities


def load_va(na_out_path, supplement_path):
    """加载预定义的attribute设置..."""

    print("加载预定义的attribute设置...")

    # 加载形容词属性之间关系
    va2attributes = dict()
    va2confidence = dict()
    with open(na_out_path, 'r') as infile:
        for line in infile:
            line = line.strip().lower()
            line = line.strip('\n')
            line = line.split('\t')
            name = line[0]
            score = int(line[1])
            word = line[2]
            confidence = int(line[3])
            if confidence > CONF.NA_LIMITH:
                va2attributes[name, word] = score
                va2confidence[name, word] = confidence
        infile.close()

    # 加载补充的形容词属性关系
    with open(supplement_path, 'r') as infile:
        for line in infile:
            line = line.strip('\n')
            line = line.split('\t')
            name = line[0]
            score = int(line[1])
            word = line[2]
            va2attributes[name, word] = score
    temp = list(va2attributes.keys())
    temp.sort(key=lambda x: va2confidence.setdefault(x, 0), reverse=True)
    with open(na_out_path, 'w') as outfile:
        for (name, word) in temp:
            score = va2attributes.setdefault((name, word), None)
            confidence = va2confidence.setdefault((name, word), 0)
            outfile.write(name + '\t' + str(score) + '\t' + word + '\t' + str(confidence) + '\n')
        outfile.close()

    return va2attributes, va2confidence


def init():
    """初始化语料库等资源"""

    print('正在进行初始化设置...')
    entities, term2entity, entity2term = load_enititiy(whole_part_path=CONF.WHOLE_PART_PATH,
                                                       entitiy_synonym_path=CONF.ENTITY_SYNONYM_PATH)
    term2attributes, attributes2term, entities = load_attribute(
        attribute_synonym_path=CONF.ATTRIBUTE_SYNONYM_PATH,
        entity_attribute_path=CONF.ENTITY_ATTRIBUTE_PATH,
        entities=entities)

    va2attributes, va2confidence = load_va(na_out_path=CONF.NA_OUT_PATH,
                                           supplement_path=CONF.SUPPLEMENT_ATTRIBUTE_SYNONYM_PATH)

    print('初始化设置成功！\n')
    return {
        'entities': entities,
        'term2entity': term2entity,
        'va2attributes': va2attributes,
        'term2attributes': term2attributes,
        'entity2term': entity2term,
        'attributes2term': attributes2term,
        'va2confidence': va2confidence
    }
