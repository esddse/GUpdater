
import os
import sys


# ========== basic setting ============
path_this_file = os.path.abspath(os.path.dirname(__file__))
path_proj_home = os.path.join(path_this_file, "..", "..")

# ============  model ==========

path_model_dir = os.path.join(path_proj_home, "model")

# ============ data ============

path_data_dir = os.path.join(path_proj_home, "data")

# vocab
path_entity2id = os.path.join(path_data_dir, "entity2id.txt")
path_relation2id = os.path.join(path_data_dir, "relation2id.txt")
path_token2id = os.path.join(path_data_dir, "token2id.txt")

# data
path_train = os.path.join(path_data_dir, "NBAtransactions_train.json")
path_valid = os.path.join(path_data_dir, "NBAtransactions_valid.json")
path_test  = os.path.join(path_data_dir, "NBAtransactions_test.json")