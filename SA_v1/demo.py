# -*- coding: utf-8 -*-

# sentence: 需要分好词 空格分隔 2<=词语数量<=30
# result: (aspect, opinion, aspect_index, opinion_index, relation, knowledge)
# knowledge: (aspect, opinion, relation, weight)

from preprocess import WordSet, WordEmbedding, KnowledgeBase
from sentiment_analysis import SentimentAnalysis


abc = SentimentAnalysis()

sentence = '外观 漂亮'
result = abc.analyze(sentence)

sentence1 = '外观 不 太 漂亮'
result1 = abc.analyze(sentence1)

sentence2 = '高 规格 的 用料 和 精致 的 做工'
result2 = abc.analyze(sentence2)

sentence3 = '炫酷 的 造型 、 充沛 的 动力 再 加上 本田 家族 运动 基因 的 传承'
result3 = abc.analyze(sentence3)
