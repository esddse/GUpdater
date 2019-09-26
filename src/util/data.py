import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import csv
import math
import json
import pickle
import re
import copy

from collections import Counter

from tqdm import tqdm
import numpy as np
import torch


# ===================== load & save =========================

def load_json(file_path):
    ''' load json file '''
    with open(file_path, "r", encoding="utf8") as f:
        data = json.load(f)
        return data

def dump_json(data, file_path):
    ''' save json file '''
    with open(file_path, "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_pkl(path):
    ''' load pkl '''
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pkl(data, path):
    ''' save pkl '''
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=2)


def load_str_lst(path):
    ''' load string list '''
    strs = []
    with open(path, "r", encoding="utf8") as f:
        for line in tqdm(f):
            strs.append(line.strip())
    return strs

def load_str_dict(path, seperator="\t", reverse=False):
    ''' load string dict '''
    dictionary, reverse_dictionay = {}, {}
    with open(path, "r", encoding="utf8") as f:
        for line in tqdm(f):
            try:
                key, value = line.strip().split(seperator)
                dictionary[key] = int(value)
                reverse_dictionay[int(value)] = key
            except:
                pass
    if reverse:
        return dictionary, reverse_dictionay, len(dictionary)
    return dictionary, len(dictionary)

def dump_str_lst(lst, path):
    ''' save string list '''
    with open(path, "w", encoding="utf8") as f:
        for string in tqdm(lst):
            f.write(string+'\n')

def load_NBAtransactions(file_path):
    ''' load data '''
    datas = load_json(file_path)
    for i in range(len(datas)):
        datas[i]["subgraph_before"] = [tuple(triple) for triple in datas[i]["subgraph_before"]]
        datas[i]["subgraph_after"]  = [tuple(triple) for triple in datas[i]["subgraph_after"]]
    return datas

# =========== task specific transformation ===========

def gen_big_graphs(datas):
    ''' generate big graphs by season '''
    big_graphs = {}
    for data in datas:
        season = data["season"]
        if season not in big_graphs:
            big_graphs[season] = set(data["subgraph_before"])
        else:
            big_graphs[season].update(data["subgraph_before"])
    return big_graphs

def get_roster(triples):
    ''' get team rosters from triples '''
    roster = {}
    for h, r, t in triples:
        if r == 1:  # <player>, (h, <player>, t) 
            if h <= 32: # team
                if h not in roster:
                    roster[h] = {t}
                else:
                    roster[h].add(t)
            else:
                if t not in roster:
                    roster[t] = {h}
                else:
                    roster[t].add(h)
    return roster


def get_k_hop_entities(triples, seeds, k=1):
    '''
        get k-hop neighbors of seeds from triples
        parameters:
            @triples: set((h,r,t), ...)
            @seeds: set(e1, e2, ...)
            @k: h-hop, int
    '''
    ret_triples, ret_entities = set(), set(seeds)
    for _ in range(k):
        new_seeds = set()
        for h, r, t in triples:
            if h in seeds or t in seeds:
                ret_triples.add((h, r, t))
                ret_entities.add(h)
                ret_entities.add(t)
                new_seeds.update([h, t])
        seeds = new_seeds
    return list(ret_entities)


def get_triples_by_entities(triples, entities):
    ''' return all triples that head & tail in entities '''
    return [(h, r, t) for h, r, t in triples if h in entities and t in entities]


def get_all_possible_triples(ref_triples, entities, relations, neg_label=0.):
    '''
        根据entities和relations生成全图，并给出对应标注
        参数:
            ref_triples: set((h,r,t), ...)
            entities:    set(e1, e2)
            relations:   set(r1, r2)
        返回:
            full_triples: [(h, r, t), ...]
            labels:       [1, 0, ...]
    '''

    full_triples = []
    labels = []
    for h in entities:
        for t in entities:
            if h == t:
                continue
            for r in relations:
                full_triples.append((h, r, t))
                if (h, r, t) in ref_triples:
                    labels.append(1.)
                else:
                    labels.append(neg_label)
    return full_triples, labels


def padding_sequence(indices, max_length, pad_idx, get_length=False):
    ''' '''
    length = len(indices) if len(indices) < max_length else max_length
    if len(indices) >= max_length:
        if get_length:
            return indices[:max_length], length
        else:
            return indices[:max_length]
    else:
        if get_length:
            return indices + [pad_idx] * (max_length - len(indices)), length
        else:
            return indices + [pad_idx] * (max_length - len(indices))

def get_adjacents_selfloop(triples, entity_size, relation_size):
    
    '''
        generate adjacent matrix for each relation type
        add new relation type <self-loop>
    '''

    # init edges
    edges_lst = [[] for _ in range(relation_size)]
    for h, r, t in triples:
        edges_lst[r].append((h, t))

    # count edge
    neighbor_counter = Counter()
    indices_pairs = []
    for edges in edges_lst:
        neighbors = [e[0] for e in edges]
        neighbor_counter.update(neighbors)
        edges.sort()
        if edges:
            row_indices, col_indices = zip(*edges)
        else:
            row_indices, col_indices = (), ()
        indices_pairs.append((row_indices, col_indices))
    # add identity (self-loop)
    neighbor_counter.update([i for i in range(entity_size)])
    identity_indices = [i for i in range(entity_size)]
    indices_pairs.append((identity_indices, identity_indices))

    # r-normalization
    adjacents = []
    for row_indices, col_indices in indices_pairs:
        if row_indices:
            weights = torch.FloatTensor([1. for i in range(len(row_indices))])
            indices = torch.LongTensor([row_indices, col_indices])
            adjacent = torch.sparse.FloatTensor(indices, weights, torch.Size([entity_size, entity_size]))
        else:
            adjacent = torch.sparse.FloatTensor(entity_size, entity_size)
        adjacents.append(adjacent)
 
    return adjacents   

def get_adjacents_shortcut(triples, text_entities, entity_size, relation_size):
    
    '''
        generate adjacent matrix for each relation type
        add new relation type <self-loop> & <shortcut>
    '''


    # init edges
    edges_lst = [[] for _ in range(relation_size)]
    for h, r, t in triples:
        edges_lst[r].append((h, t))

    # count edge
    neighbor_counter = Counter()
    indices_pairs = []
    for edges in edges_lst:
        neighbors = [e[0] for e in edges]
        neighbor_counter.update(neighbors)
        edges.sort()
        if edges:
            row_indices, col_indices = zip(*edges)
        else:
            row_indices, col_indices = (), ()
        indices_pairs.append((row_indices, col_indices))
    # add identity
    neighbor_counter.update([i for i in range(entity_size)])
    identity_indices = [i for i in range(entity_size)]
    indices_pairs.append((identity_indices, identity_indices))
    # add text
    text_edges = [(ett1, ett2) for ett1 in text_entities for ett2 in text_entities if ett1 != ett2]
    text_edges.sort()
    if not text_edges:
        row_indices, col_indices = [], []
    else:
        row_indices, col_indices = zip(*text_edges)
    indices_pairs.append((row_indices, col_indices))

    # r-normalization
    adjacents = []
    for row_indices, col_indices in indices_pairs:
        if row_indices:
            weights = torch.FloatTensor([1. for i in range(len(row_indices))])
            indices = torch.LongTensor([row_indices, col_indices])
            adjacent = torch.sparse.FloatTensor(indices, weights, torch.Size([entity_size, entity_size]))
        else:
            adjacent = torch.sparse.FloatTensor(entity_size, entity_size)
        adjacents.append(adjacent)
 
    return adjacents    

def get_ie_label(text_entities, event, roster):
    
    '''
        get shortcut labels
    '''

    def judge_edge(ett1, ett2, event, roster):
        if event in ["free_agency", "draft"]:
            return "add"
        elif event in ["released", "retirement", "d_league", "overseas"]:
            return "del"
        elif event in ["head_coach", "general_manager"]:
            return "other"
        else:  # trade
            if ett1 in roster and ett2 not in roster: # ett1: team, ett2: player
                if ett2 in roster[ett1]:
                    return "del"
                else:
                    return "add"
            elif ett1 not in roster and ett2 in roster: # ett1: player, ett2: team
                if ett1 in roster[ett2]:
                    return "del"
                else:
                    return "add"
            else: # both team or both player
                return "other"

    label_to_idx = {"add":0, "del":1,"other":2}
    heads, tails, labels = [], [], []
    for ett1 in text_entities:
        for ett2 in text_entities:
            if ett1 == ett2:
                continue
            label = judge_edge(ett1, ett2, event, roster)
            label = label_to_idx[label]
            heads.append(ett1)
            tails.append(ett2)
            labels.append(label)
    return heads, tails, labels

# ==================== metrics =======================

def my_pos_avg(y_pred, y_true):
    return torch.mean((y_true + 1.) * y_pred / 2.0)
def my_neg_avg(y_pred, y_true):
    return torch.mean((1. - y_true) * y_pred / 2.0)
def compute_acc(y_pred, y_true):
    return torch.sum(torch.abs(y_pred - y_true) < 0.5).item()
def F1(precision, recall):
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.

# ===================== main =========================

def main():
    pass

if __name__ == '__main__':
    main()