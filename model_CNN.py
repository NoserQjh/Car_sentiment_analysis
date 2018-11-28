# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable


class SentenceEmbedding(nn.Module):
    def __init__(self, word_embedding, position1_embedding, position2_embedding,
                 in_channels, out_channels, kernel_size):
        super(SentenceEmbedding, self).__init__()
        self.word_embedding = word_embedding
        self.position1_embedding = position1_embedding
        self.position2_embedding = position2_embedding
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.relu = nn.ReLU()
    
    def forward(self, sentence, position1, position2, word_index):
        # sentence: (n_sentences, sequence_length)
        # position1: (n_sentences, sequence_length)
        # position2: (n_sentences, sequence_length)
        # word_index: (n_sentences, 2)
        
        # embedding
        word_embed = self.word_embedding(sentence) # (n_sentences, sequence_length, word_dim)
        pos1_embed = self.position1_embedding(position1) # (n_sentences, sequence_length, position_dim)
        pos2_embed = self.position2_embedding(position2) # (n_sentences, sequence_length, position_dim)
        snt_embed = torch.cat((word_embed, pos1_embed, pos2_embed), 2) # (n_sentences, sequence_length, word_dim+2*position_dim)
        snt_embed = snt_embed.unsqueeze(1) # (n_sentences, 1, sequence_length, word_dim+2*position_dim)
        
        # convolution
        out = self.cnn(snt_embed) # (n_sentences, out_channels, out_sequence_length, 1)
        out = self.relu(out) # (n_sentences, out_channels, out_sequence_length, 1)
        
        # piecewice max pooling
        n_sentences = out.shape[0]
        out_channels = out.shape[1]
        snt_vector = Variable(torch.zeros(n_sentences, 3 * out_channels))
        for i in range(n_sentences):
            idx1 = word_index[i, 0].item() - 1
            idx2 = word_index[i, 1].item() - 1
            pool1, _ = torch.max(out[i, :, :idx1+1, :], dim=1) # (out_channels, 1)
            if idx1+1 < idx2:
                pool2, _ = torch.max(out[i, :, idx1+1:idx2, :], dim=1) # (out_channels, 1)
            else:
                pool2 = Variable(torch.zeros(out_channels, 1))
            pool3, _ = torch.max(out[i, :, idx2:, :], dim=1) # (out_channels, 1)
            pool = torch.cat((pool1, pool2, pool3), 1) # (out_channels, 3)
            pool = pool.view(1, -1) # (1, 3*out_channels)
            snt_vector[i] = pool
        
        return snt_vector # (n_sentences, 3*out_channels)


class Classifier(nn.Module):
    def __init__(self, input_dim, n_relations, hidden_size_cls, dropout_prob):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_size_cls)
        self.linear2 = nn.Linear(hidden_size_cls, n_relations)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, snt_vector):
        # snt_vector: (n_sentences, 3*out_channels)

        out = self.dropout(snt_vector) # (n_sentences, 3*out_channels)
        out = self.linear1(out) # (n_sentences, hidden_size_cls)
        out = self.relu(out)
        out = self.dropout(out)
        predict = self.linear2(out) # (n_sentences, n_relations)
        
        return predict # (n_sentences, n_relations)


class Model(nn.Module):
    def __init__(self, vocab_size, word_dim, position_size, position_dim,
                 in_channels, out_channels, kernel_size,
                 hidden_size_cls, n_relations, dropout_prob):
        super(Model, self).__init__()

        self.word_embedding = nn.Embedding(vocab_size, word_dim)
        self.position1_embedding = nn.Embedding(position_size, position_dim)
        self.position2_embedding = nn.Embedding(position_size, position_dim)
        
        self.sentence_embedding = SentenceEmbedding(self.word_embedding, self.position1_embedding, self.position2_embedding,
                                                    in_channels, out_channels, kernel_size)
        self.classifier = Classifier(3*out_channels, n_relations, hidden_size_cls, dropout_prob)
    
    def forward(self, sentence, position1, position2, word_index):
        snt_vector = self.sentence_embedding(sentence, position1, position2, word_index)
        predict = self.classifier(snt_vector)
        return predict