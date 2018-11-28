# -*- coding: utf-8 -*-

import pickle
import numpy as np


class WordSet:
    def __init__(self, n_aspects, n_opinions, aspect, opinion):
        self.n_aspects = n_aspects
        self.n_opinions = n_opinions
        self.aspect = aspect
        self.opinion = opinion


class WordEmbedding:
    def __init__(self, vocab_size, word_dim, word_to_index, index_to_word, matrix):
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
        self.matrix = matrix


class KnowledgeBase:
    def __init__(self, n_triples, triple):
        self.n_triples = n_triples
        self.triple = triple # (n_triples, 3)


def load_word_set(filename_ent, filename_att, filename_des):
    print('Loading word set from %s...' % (filename_ent))
    print('Loading word set from %s...' % (filename_att))
    print('Loading word set from %s...' % (filename_des))
    
    aspect = set()
    opinion = set()
    file = open(filename_ent, 'r', encoding='utf-8')
    for line in file:
        word = line.strip()
        if word:
            aspect.add(word)
    file.close()
    file = open(filename_att, 'r', encoding='utf-8')
    for line in file:
        word = line.strip()
        if word:
            aspect.add(word)
    file.close()
    file = open(filename_des, 'r', encoding='utf-8')
    for line in file:
        word = line.strip()
        if word:
            opinion.add(word)
    file.close()
    
    n_aspects = len(aspect)
    n_opinions = len(opinion)
    
    print('n_aspects = %d' % (n_aspects))
    print('n_opinions = %d' % (n_opinions))
    return WordSet(n_aspects, n_opinions, aspect, opinion)


def load_word_embedding(filename):
    print('Loading word embedding from %s...' % (filename))
    
    file = open(filename, 'r', encoding='utf-8')
    # get size and dimension
    line = file.readline().strip()
    vocab_size = int(line.split(' ')[0]) + 1
    word_dim = int(line.split(' ')[1])
    matrix = np.zeros([vocab_size, word_dim], dtype=np.float32)
    
    # unknown word embedding
    word_to_index = {}
    index_to_word = {}
    index = 0
    word_to_index['UNK'] = index
    index_to_word[index] = 'UNK'
    index += 1
    
    # word embedding
    for line in file:
        lsp = line.strip().split(' ')
        if len(lsp) == word_dim + 1:
            word = lsp[0]
            vector = np.array(lsp[1:], dtype=np.float32)
            word_to_index[word] = index
            index_to_word[index] = word
            matrix[index] = vector
            index += 1
    file.close()
    
    print('vocab_size = %d' % (vocab_size))
    print('word_dim = %d' % (word_dim))
    return WordEmbedding(vocab_size, word_dim, word_to_index, index_to_word, matrix)


def load_knowledge_base(filename, word_to_index, rel_to_index):
    print('Loading knowledge base from %s...' % (filename))
    
    triple = []
    file = open(filename, 'r', encoding='utf-8')
    for line in file:
        lsp = line.strip().split('\t')
        if (lsp[0] in word_to_index) and (lsp[1] in word_to_index) and (lsp[2] in rel_to_index):
            t = [word_to_index[lsp[0]], word_to_index[lsp[1]], rel_to_index[lsp[2]]]
            triple.append(t)
    file.close()
    
    n_triples = len(triple)
    triple = np.array(triple, dtype=np.int32)
    rand_idx = np.random.permutation(n_triples)
    triple = triple[rand_idx]
    
    print('n_triples = %d' % (n_triples))
    return KnowledgeBase(n_triples, triple)


if __name__ == '__main__':
    '''
    word_set = load_word_set('data/word_entity.txt', 'data/word_attribute.txt', 'data/word_description.txt')
    with open('data/word_set.pkl', 'wb') as f:
        pickle.dump(word_set, f)
    '''
    '''
    # word embedding
    word_embedding = load_word_embedding('data/vectors_50d.txt')
    with open('data/word_embedding.pkl', 'wb') as f:
        pickle.dump(word_embedding, f)
    '''
    '''
    # word to index
    with open('data/word_embedding.pkl', 'rb') as f:
        word_embedding = pickle.load(f)
    word_to_index = word_embedding.word_to_index
    
    # relation to index
    rel_to_index = {'POS':0, 'NEU':1, 'NEG':2}
    '''
    '''
    # knowledge base
    knowledge_base = load_knowledge_base('data/triple_SP.txt', word_to_index, rel_to_index)
    with open('data/knowledge_base.pkl', 'wb') as f:
        pickle.dump(knowledge_base, f)
    '''