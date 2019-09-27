import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from util.path import *
from util.data import *

class GUpdaterConfig():

    def __init__(self):

        # basic
        self.model_name = "GUpdater+label"
        self.graph_embedding_dim = 128
        self.hidden_dim = 256
        self.batch_size = 1
        self.epoch_num = 100
        self.report_step_num = 100
        self.dropout_rate = 0.5
        self.learning_rate = 1e-3
        self.min_learning_rate = 1e-4
        self.decay_factor = 0.3
        self.patience = 2

        # task specific
        self.text_max_length = 120
        self.pad_idx = 6691
        self.basis_num = 2
        self.shortcut_setting = "labeled" # "no" or "unified" or "labeled"
        self.use_text = True
        self.k_hop = 1

        # train
        self.gpu_id = "0"

        # test
        self.to_test = ["1hop_subgraph", "1hop_added", "1hop_deleted", "1hop_unchanged", 
                        "text_subgraph", "text_added", "text_deleted", "text_unchanged"]

        # vocab
        self.entity_path = path_entity2id
        self.relation_path = path_relation2id
        self.token_path = path_token2id

        # init
        self.init()


    def init(self):
        ''' additional configuration '''

        # vocab
        self.entity2id,   self.id2entity,   self.entity_size   = load_str_dict(self.entity_path, reverse=True)
        self.relation2id, self.id2relation, self.relation_size = load_str_dict(self.relation_path, reverse=True)
        self.token2id,    self.id2token,    self.token_size    = load_str_dict(self.token_path, reverse=True)


        # extra adjacent matrix number
        if self.shortcut_setting == "no":
            self.add_adj_size = 1  # selfloop
        elif self.shortcut_setting == "unified":
            self.add_adj_size = 2  
        elif self.shortcut_setting == "labeled":
            self.add_adj_size = 4 

