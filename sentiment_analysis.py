# -*- coding: utf-8 -*-

import copy
import pickle
import numpy as np
import torch
from torch.autograd import Variable

from preprocess import WordSet, WordEmbedding, KnowledgeBase


MAX_PAIR_DIST = 6
SENTENCE_LENGTH = 30
PADDING_LENGTH = 1
N_CANDIDATES = 10


class Data:
    def __init__(self, n_sentences, sentence, position1, position2, word_index, key_word):
        self.n_sentences = n_sentences
        self.sentence = sentence # (n_sentences, sequence_length)
        self.position1 = position1 # (n_sentences, sequence_length)
        self.position2 = position2 # (n_sentences, sequence_length)
        self.word_index = word_index # (n_sentences, 2)
        self.key_word = key_word # (n_sentences, 2)


class SentimentAnalysis:
    def __init__(self):
        # load data
        print('Loading data...')
        with open('data/word_set.pkl', 'rb') as f:
            self.word_set = pickle.load(f)
        with open('data/word_embedding.pkl', 'rb') as f:
            self.word_embedding = pickle.load(f)
        with open('data/knowledge_base.pkl', 'rb') as f:
            self.knowledge_base = pickle.load(f)
        self.rel_to_index = {'POS':0, 'NEU':1, 'NEG':2, 'NA':3}
        self.index_to_rel = {0:'POS', 1:'NEU', 2:'NEG', 3:'NA'}
        print('Done')
        
        # load model
        print('Loading model...')
        self.model_PAIR = torch.load('model/model_CNN_PAIR_18')
        self.model_SENT = torch.load('model/model_MNKG_SENT_19')
        self.model_PAIR.eval()
        self.model_SENT.eval()
        print('Done')
    
    
    def find_potential_pair(self, sentence):
        sentence = sentence.strip()
        words = sentence.split(' ')
        
        aspects = []
        opinions = []
        for idx, word in enumerate(words):
            if word in self.word_set.aspect:
                aspects.append((word, idx))
            if word in self.word_set.opinion:
                opinions.append((word, idx))
        
        quintuplets = set() # (aspect, opinion, aspect_index, opinion_index, sentence)
        for aspect_idx in aspects:
            for opinion_idx in opinions:
                if abs(aspect_idx[1] - opinion_idx[1]) <= MAX_PAIR_DIST:
                    q = (aspect_idx[0], opinion_idx[0], aspect_idx[1], opinion_idx[1], sentence)
                    quintuplets.add(q)
        
        return quintuplets
    
    
    def preprocess(self, quintuplets):
        sentence = []
        position1 = []
        position2 = []
        word_index = []
        key_word = []
        for q in quintuplets:
            word1 = q[0]
            word2 = q[1]
            pos1 = q[2]
            pos2 = q[3]
            snt = q[4].split(' ')
            # unknow word
            if word1 not in self.word_embedding.word_to_index:
                word1 = 'UNK'
            if word2 not in self.word_embedding.word_to_index:
                word2 = 'UNK'
            for i in range(len(snt)):
                if snt[i] not in self.word_embedding.word_to_index:
                    snt[i] = 'UNK'
            # to sequence
            snt_seq = [self.word_embedding.word_to_index[word] for word in snt]
            pos1_seq = [(i - pos1 + SENTENCE_LENGTH) for i in range(len(snt))]
            pos2_seq = [(i - pos2 + SENTENCE_LENGTH) for i in range(len(snt))]
            # padding: sequence_length = SENTENCE_LENGTH + 2 * PADDING_LENGTH
            snt_seq = [0] * PADDING_LENGTH + snt_seq + [0] * (SENTENCE_LENGTH - len(snt_seq)) + [0] * PADDING_LENGTH
            pos1_seq =  [0] * PADDING_LENGTH + pos1_seq + [0] * (SENTENCE_LENGTH - len(pos1_seq)) + [0] * PADDING_LENGTH
            pos2_seq =  [0] * PADDING_LENGTH + pos2_seq + [0] * (SENTENCE_LENGTH - len(pos2_seq)) + [0] * PADDING_LENGTH
            # word index
            wi = sorted([pos1 + PADDING_LENGTH, pos2 + PADDING_LENGTH])
            # key word
            kw = [self.word_embedding.word_to_index[word1], self.word_embedding.word_to_index[word2]]
            
            sentence.append(snt_seq)
            position1.append(pos1_seq)
            position2.append(pos2_seq)
            word_index.append(wi)
            key_word.append(kw)
        
        n_sentences = len(sentence)
        sentence = np.array(sentence, dtype=np.int32)
        position1 = np.array(position1, dtype=np.int32)
        position2 = np.array(position2, dtype=np.int32)
        word_index = np.array(word_index, dtype=np.int32)
        key_word = np.array(key_word, dtype=np.int32)
        
        return Data(n_sentences, sentence, position1, position2, word_index, key_word)
    
    
    def set_kw_to_sm(self, data):
        kw_to_sm = {}
        for i in range(data.n_sentences):
            kw = (data.key_word[i, 0], data.key_word[i, 1])
            cos_sim = np.zeros([self.knowledge_base.n_triples, 2], dtype=np.float32)
            for j in range(self.knowledge_base.n_triples):
                for k in range(2):
                    f1 = self.word_embedding.matrix[kw[k]]
                    f2 = self.word_embedding.matrix[self.knowledge_base.triple[j, k]]
                    cos_sim[j, k] = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-5)
            cos_sim = np.sum(cos_sim, axis=1)
            sort_idx = np.argsort(cos_sim)
            sort_idx = sort_idx[::-1]
            kw_to_sm[kw] = sort_idx
        self.model_SENT.knowledge_retrieval.kw_to_sm = copy.deepcopy(kw_to_sm)
    
    
    def classify(self, data):
        sentence = Variable(torch.LongTensor(data.sentence))
        position1 = Variable(torch.LongTensor(data.position1))
        position2 = Variable(torch.LongTensor(data.position2))
        word_index = Variable(torch.LongTensor(data.word_index))
        key_word = Variable(torch.LongTensor(data.key_word))
        
        predict_PAIR = self.model_PAIR(sentence, position1, position2, word_index)
        _, predict_PAIR = torch.max(predict_PAIR, dim=1)
        predict_PAIR = predict_PAIR.data.numpy() # (n_sentences)
        
        predict_SENT, candidate, weight = self.model_SENT(sentence, position1, position2, key_word)
        _, predict_SENT = torch.max(predict_SENT, dim=1)
        predict_SENT = predict_SENT.data.numpy() # (n_sentences)
        candidate = candidate.data.numpy() # (n_sentences, n_candidates, 3)
        weight = weight.data.numpy()[:, :, 0] # (n_sentences, n_candidates)
        
        predict = predict_SENT
        predict[predict_PAIR == 0] = 3 # (n_sentences)
        
        return predict, candidate, weight
        
    def analyze(self, sentence):
        print('Analyzing...')
        result = []
        
        # check input sentence
        sentence = sentence.strip()
        words = sentence.split(' ')
        if (len(words) < 2) or (len(words) > SENTENCE_LENGTH):
            print('Invalid input: %s' % (sentence))
            print('Done')
            return result
        
        quintuplets = self.find_potential_pair(sentence)
        data = self.preprocess(quintuplets)
        self.set_kw_to_sm(data)
        predict, candidate, weight = self.classify(data)
        
        for i in range(data.n_sentences):
            if predict[i] == 3:
                continue
            # key word
            word1 = self.word_embedding.index_to_word[data.key_word[i, 0]]
            word2 = self.word_embedding.index_to_word[data.key_word[i, 1]]
            # word index
            pos1 = data.word_index[i, 0]
            pos2 = data.word_index[i, 1]
            if data.position1[i, pos1] != SENTENCE_LENGTH:
                pos1 = data.word_index[i, 1]
                pos2 = data.word_index[i, 0]
            pos1 = pos1 - PADDING_LENGTH
            pos2 = pos2 - PADDING_LENGTH
            # relation
            rel = self.index_to_rel[predict[i]]
            # candidate and weight
            candidates = []
            candidate[i]
            for j in range(N_CANDIDATES):
                w1 = self.word_embedding.index_to_word[candidate[i, j, 0]]
                w2 = self.word_embedding.index_to_word[candidate[i, j, 1]]
                r = self.index_to_rel[candidate[i, j, 2]]
                w = weight[i, j]
                candidates.append((w1, w2, r, w))
            result.append((word1, word2, pos1, pos2, rel, candidates))
        
        print('Done')
        return result
