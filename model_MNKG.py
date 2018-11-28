# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable


class SentenceEmbedding(nn.Module):
    def __init__(self, word_embedding, position1_embedding, position2_embedding,
                 wp_dim, hidden_size_att):
        super(SentenceEmbedding, self).__init__()
        self.word_embedding = word_embedding
        self.position1_embedding = position1_embedding
        self.position2_embedding = position2_embedding
        self.linear1 = nn.Linear(2*wp_dim, hidden_size_att)
        self.linear2 = nn.Linear(hidden_size_att, 1)
        self.relu = nn.ReLU()
    
    def forward(self, sentence, position1, position2):
        # sentence: (n_sentences, sequence_length)
        # position1: (n_sentences, sequence_length)
        # position2: (n_sentences, sequence_length)
        
        # embedding
        word_embed = self.word_embedding(sentence) # (n_sentences, sequence_length, word_dim)
        pos1_embed = self.position1_embedding(position1) # (n_sentences, sequence_length, position_dim)
        pos2_embed = self.position2_embedding(position2) # (n_sentences, sequence_length, position_dim)
        snt_embed = torch.cat((word_embed, pos1_embed, pos2_embed), 2) # (n_sentences, sequence_length, word_dim+2*position_dim)
        
        n_sentences = snt_embed.shape[0]
        seq_len = snt_embed.shape[1]
        wp_dim = snt_embed.shape[2]
        
        query = Variable(torch.zeros(n_sentences, wp_dim)) # (n_sentences, word_dim+2*position_dim)
        for i in range(n_sentences):
            idx = 0
            for j in range(seq_len):
                if position1[i, j] == 30:
                    idx = j
            query[i, :] = snt_embed[i, idx, :]
        
        for h in range(6):
            q = query.unsqueeze(1) # (n_sentences, 1, word_dim+2*position_dim)
            q = q.repeat(1, seq_len, 1) # (n_sentences, sequence_length, word_dim+2*position_dim)
            feature = torch.cat((snt_embed, q), 2) # (n_sentences, sequence_length, 2*(word_dim+2*position_dim))
            
            out = self.linear1(feature) # (n_sentences, sequence_length, hidden_size_att)
            out = self.relu(out)
            weight = self.linear2(out) # (n_sentences, sequence_length, 1)
            
            # weight average
            weight = weight.squeeze(2) # (n_sentences, sequence_length)
            weight = torch.exp(weight)
            summary = torch.sum(weight, dim=1) # (n_sentences)
            summary = summary.unsqueeze(1) # (n_sentences, 1)
            summary = summary.repeat(1, seq_len) # (n_sentences, sequence_length)
            weight = weight / summary; # (n_sentences, sequence_length)
            weight = weight.unsqueeze(2) # (n_sentences, sequence_length, 1)
            weight = weight.repeat(1, 1, wp_dim) # (n_sentences, sequence_length, word_dim+2*position_dim)
            temp = snt_embed * weight # (n_sentences, sequence_length, word_dim+2*position_dim)
            query = torch.sum(temp, dim=1) # (n_sentences, word_dim+2*position_dim)
        snt_vector = query
        
        return snt_vector # (n_sentences, word_dim+2*position_dim)


class KnowledgeRetrieval(nn.Module):
    def __init__(self, n_triples, n_candidates):
        super(KnowledgeRetrieval, self).__init__()
        self.n_triples = n_triples
        self.n_candidates = n_candidates
        self.triple = Variable(torch.LongTensor(n_triples, 3), requires_grad=False)
        self.kw_to_sm = {}

    def forward(self, key_word):
        # key_word: (n_sentences, 2)
        
        n_sentences = key_word.shape[0]
        candidate = Variable(torch.LongTensor(n_sentences, self.n_candidates, 3))
        for i in range(n_sentences):
            kw = (int(key_word[i, 0]), int(key_word[i, 1]))
            candidate_idx = self.kw_to_sm[kw][:self.n_candidates]
            candidate[i] = self.triple[candidate_idx, :]
        
        return candidate # (n_sentences, n_candidates, 3)


class KnowledgeEmbedding(nn.Module):
    def __init__(self, word_embedding, relation_embedding):
        super(KnowledgeEmbedding, self).__init__()
        self.word_embedding = word_embedding
        self.relation_embedding = relation_embedding
    
    def forward(self, candidate):
        # candidate: (n_sentences, n_candidates, 3)

        n_sentences = candidate.shape[0]
        n_candidates = candidate.shape[1]
        
        word_embed = self.word_embedding(candidate[:, :, :2]) # (n_sentences, n_candidates, 2, word_dim)
        word_embed = word_embed.view(n_sentences, n_candidates, -1) # (n_sentences, n_candidates, 2*word_dim)
        rel_embed = self.relation_embedding(candidate[:, :, 2]) # (n_sentences, n_candidates, relation_dim)
        mem_vector = torch.cat((word_embed, rel_embed), 2) # (n_sentences, n_candidates, 2*word_dim+relation_dim)
        
        return mem_vector # (n_sentences, n_candidates, 2*word_dim+relation_dim)


class QueryGeneration(nn.Module):
    def __init__(self, input_dim, query_dim):
        super(QueryGeneration, self).__init__()
        self.linear1 = nn.Linear(input_dim, query_dim)
        self.relu = nn.ReLU()
    
    def forward(self, snt_vector):
        # snt_vector: (n_sentences, input_dim)

        out = self.linear1(snt_vector) # (n_sentences, query_dim)
        query = self.relu(out)

        return query # (n_sentences, query_dim)


class KnowledgeSelection(nn.Module):
    def __init__(self, input_dim, hidden_size_ks):
        super(KnowledgeSelection, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_size_ks)
        self.linear2 = nn.Linear(hidden_size_ks, 1)
        self.relu = nn.ReLU()
    
    def forward(self, mem_vector, query):
        # mem_vector: (n_sentences, n_candidates, 2*word_dim+relation_dim)
        # query: (n_sentences, query_dim)
        
        n_sentences = mem_vector.shape[0]
        n_candidates = mem_vector.shape[1]
        mem_dim = mem_vector.shape[2]
        
        # attention weight
        query = query.unsqueeze(1) # (n_sentences, 1, query_dim)
        query = query.repeat(1, n_candidates, 1) # (n_sentences, n_candidates, query_dim)
        feature = torch.cat((mem_vector, query), 2) # (n_sentences, n_candidates, 2*word_dim+relation_dim+query_dim)
        out = self.linear1(feature) # (n_sentences, n_candidates, hidden_size_ks)
        out = self.relu(out)
        weight = self.linear2(out) # (n_sentences, n_candidates, 1)
        
        # weight average
        weight = weight.squeeze(2) # (n_sentences, n_candidates)
        weight = torch.exp(weight)
        summary = torch.sum(weight, dim=1) # (n_sentences)
        summary = summary.unsqueeze(1) # (n_sentences, 1)
        summary = summary.repeat(1, n_candidates) # (n_sentences, n_candidates)
        weight = weight / summary; # (n_sentences, n_candidates)
        weight = weight.unsqueeze(2) # (n_sentences, n_candidates, 1)
        weight = weight.repeat(1, 1, mem_dim) # (n_sentences, n_candidates, 2*word_dim+relation_dim)
        temp = mem_vector * weight # (n_sentences, n_candidates, 2*word_dim+relation_dim)
        know_vector = torch.sum(temp, dim=1) # (n_sentences, 2*word_dim+relation_dim)
        
        return know_vector, weight # (n_sentences, 2*word_dim+relation_dim)


class Classifier(nn.Module):
    def __init__(self, input_dim, n_relations, hidden_size_cls, dropout_prob):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_size_cls)
        self.linear2 = nn.Linear(hidden_size_cls, n_relations)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, snt_vector, know_vector):
        # snt_vector: (n_sentences, word_dim+2*position_dim)
        # know_vector: (n_sentences, 2*word_dim+relation_dim)
        
        feature = torch.cat((snt_vector, know_vector), 1) # (n_sentences, word_dim+2*position_dim+2*word_dim+relation_dim)
        out = self.dropout(feature) # (n_sentences, word_dim+2*position_dim+2*word_dim+relation_dim)
        out = self.linear1(out) # (n_sentences, hidden_size_cls)
        out = self.relu(out)
        out = self.dropout(out)
        predict = self.linear2(out) # (n_sentences, n_relations)
        
        return predict # (n_sentences, n_relations)


class Model(nn.Module):
    def __init__(self, vocab_size, word_dim, position_size, position_dim, relation_size, relation_dim,
                 hidden_size_att,
                 n_triples, n_candidates,
                 query_dim,
                 hidden_size_ks,
                 hidden_size_cls, n_relations, dropout_prob):
        super(Model, self).__init__()

        self.word_embedding = nn.Embedding(vocab_size, word_dim)
        self.position1_embedding = nn.Embedding(position_size, position_dim)
        self.position2_embedding = nn.Embedding(position_size, position_dim)
        self.relation_embedding = nn.Embedding(relation_size, relation_dim)
        
        self.sentence_embedding = SentenceEmbedding(self.word_embedding, self.position1_embedding, self.position2_embedding,
                                                    word_dim+2*position_dim, hidden_size_att)
        self.knowledge_retrieval = KnowledgeRetrieval(n_triples, n_candidates)
        self.knowledge_embedding = KnowledgeEmbedding(self.word_embedding, self.relation_embedding)
        self.query_generation = QueryGeneration(word_dim+2*position_dim, query_dim)
        self.knowledge_selection = KnowledgeSelection(2*word_dim+relation_dim+query_dim, hidden_size_ks)
        self.classifier = Classifier(word_dim+2*position_dim+2*word_dim+relation_dim, n_relations, hidden_size_cls, dropout_prob)
    
    def forward(self, sentence, position1, position2, key_word):
        snt_vector = self.sentence_embedding(sentence, position1, position2) # (n_sentences, word_dim+2*position_dim)
        candidate = self.knowledge_retrieval(key_word) # (n_sentences, n_candidates, 3)
        mem_vector = self.knowledge_embedding(candidate) # (n_sentences, n_candidates, 2*word_dim+relation_dim)
        query = self.query_generation(snt_vector) # (n_sentences, query_dim)
        know_vector, weight = self.knowledge_selection(mem_vector, query) # (n_sentences, 2*word_dim+relation_dim)
        predict = self.classifier(snt_vector, know_vector) # (n_sentences, n_relations)
        return predict, candidate, weight