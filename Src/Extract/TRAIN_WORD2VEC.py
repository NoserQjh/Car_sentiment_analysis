# coding=utf-8
# encoding=utf8

import time

import gensim

from Src.Extract.CONFIG import CONF

if __name__ == '__main__':

    '''t = time.time()
    more_sentences = []
    with open(CONF.REVIEWS_CWS_PATH, 'r') as infile:
        for line in infile:
            line = unicode(line, "utf-8")
            line = line.split('\t')
            more_sentences.append(line)
    print(time.time() - t)

    t = time.time()
    model = gensim.models.Word2Vec.load(CONF.WORD2VEC_PATH)
    print(time.time() - t)

    t = time.time()
    model.build_vocab(more_sentences, update=True)
    print(time.time() - t)

    t = time.time()
    model.train(more_sentences, total_examples=model.corpus_count, epochs=model.iter)
    print(time.time() - t)

    model.save(CONF.WORD2VEC_PATH_ADD)'''
    model = gensim.models.Word2Vec.load(CONF.WORD2VEC_PATH)
    print(model.similarity(u'',u''))
    model = gensim.models.Word2Vec.load(CONF.WORD2VEC_PATH_ADD)
    print(model.similarity(u'', u''))
    model = gensim.models.Word2Vec.load(CONF.WORD2VEC_PATH_NEW)
    print(model.similarity('', ''))
