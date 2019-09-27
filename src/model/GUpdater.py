
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".")))

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from layers.TextRelationalGraphAttention import TextRelationalGraphAttention
from layers.GRUEncoder import GRUEncoder

class GUpdater(nn.Module):
    def __init__(self, config):
        super(GUpdater, self).__init__()
        self.config = config

        self.token_embedding    = nn.Embedding(config.token_size, config.graph_embedding_dim)
        self.relation_embedding = nn.Embedding(config.relation_size, config.graph_embedding_dim) 

        # text
        self.gru = GRUEncoder(self.config.graph_embedding_dim, self.config.hidden_dim)
        # labeled sortcuts
        if config.shortcut_setting == "labeled":
            self.label_mlp = nn.Sequential(
                                nn.Dropout(config.dropout_rate),
                                nn.Linear(2 * config.hidden_dim, config.hidden_dim),
                                nn.LayerNorm(config.hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(config.hidden_dim, 3)
                          )
        # encoder
        self.r_gat = TextRelationalGraphAttention(self.config.graph_embedding_dim, self.config.hidden_dim, self.config.hidden_dim,
                                                   self.config.relation_size+self.config.add_adj_size, basis_num=self.config.basis_num,
                                                   activation="relu", use_text=config.use_text)

        self.drop  = nn.Dropout(config.dropout_rate)
        self.dense = nn.Linear(config.hidden_dim, config.graph_embedding_dim)

    def forward(self, input_dict):

        # unpack inputs
        entity_indices = input_dict["entity_indices"]
        text_indices   = input_dict["text_indices"]
        text_lengths   = input_dict["text_lengths"]
        triple_head_indices     = input_dict["triple_head_indices"]
        triple_relation_indices = input_dict["triple_relation_indices"]
        triple_tail_indices     = input_dict["triple_tail_indices"]
        adjacents = [input_dict["adjacent_%d" % i] for i in range(self.config.relation_size + self.config.add_adj_size) if ("adjacent_%d" % i) in input_dict]
        if self.config.shortcut_setting == "labeled":
            label_head_indices = input_dict["label_head_indices"]
            label_tail_indices = input_dict["label_tail_indices"]

        # embedding and encoding
        entity_embeddings = self.token_embedding(entity_indices)
        text_embeddings   = self.token_embedding(text_indices)
        text_encodings    = self.gru(text_embeddings, text_lengths)

        # shortcuts labeling
        if self.config.shortcut_setting == "labeled":
            adj_to_use = [i for i in range(len(adjacents))]
            label_text_entity_embeddings = self.r_gat([entity_embeddings, text_encodings, adj_to_use] + adjacents)
            label_head_embeddings = F.embedding(label_head_indices, label_text_entity_embeddings)
            label_tail_embeddings = F.embedding(label_tail_indices, label_text_entity_embeddings)
            label_head_tail = torch.cat((label_head_embeddings, label_tail_embeddings), dim=-1)
            # mlp 
            label_score = self.label_mlp(label_head_tail)
            label_adjacents = self.generate_adjacents_label(label_head_indices, label_tail_indices, label_score)
            adjacents = adjacents[:-1] + label_adjacents
            adj_to_use = [i for i in range(len(adjacents))]
        else:
            adj_to_use = [i for i in range(len(adjacents))]

        # R-GAT fusion
        fusioned_entity_embeddings = self.r_gat([entity_embeddings, text_encodings, adj_to_use] + adjacents)
        fusioned_entity_embeddings = self.dense(fusioned_entity_embeddings)

        # DistMult decode
        triple_heads = F.embedding(triple_head_indices, fusioned_entity_embeddings)
        triple_tails = F.embedding(triple_tail_indices, fusioned_entity_embeddings)
        triple_relations = self.relation_embedding(triple_relation_indices)

        # score
        score = triple_heads * triple_relations * triple_tails
        score = torch.sum(score, dim=-1)
        score = torch.sigmoid(score)
        if self.config.shortcut_setting == "labeled":
            return score, label_score
        return score

    def generate_adjacents_label(self, head_indices, tail_indices, label_score):
        ''' generate adjacent matrices according to label_score '''
        label_arange = torch.arange(label_score.size(0)).cuda()
        label_class  = torch.argmax(label_score, dim=-1)

        label_adjacents = []
        for c in [0, 1, 2]:
            mask = (label_class == c)
            head_indices_c = torch.index_select(head_indices, 0, torch.masked_select(label_arange, mask))
            tail_indices_c = torch.index_select(tail_indices, 0, torch.masked_select(label_arange, mask))
            if len(head_indices_c):
                weights = torch.ones_like(head_indices_c).float().cuda()
                head_indices_c = torch.unsqueeze(head_indices_c, 0)
                tail_indices_c = torch.unsqueeze(tail_indices_c, 0)
                indices  = torch.cat((head_indices_c, tail_indices_c), dim=0) 
                adjacent = torch.sparse.FloatTensor(indices, weights, torch.Size([self.config.entity_size, self.config.entity_size])).cuda() 
            else:
                adjacent = torch.sparse.FloatTensor(self.config.entity_size, self.config.entity_size).cuda()
            label_adjacents.append(adjacent)
        return label_adjacents