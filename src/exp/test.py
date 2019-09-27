import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import math
import random

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from util.path import *
from util.data import *
from config.config import GUpdaterConfig
from model.GUpdater import GUpdater


def data_generator(datas, big_graphs, rosters, config, shuffle=True, k_hop=0):

    batch_size = config.batch_size
    data_size = len(datas)
    batch_num = math.ceil(data_size / batch_size)


    if shuffle:
        random.shuffle(datas)
    for batch_idx in range(batch_num):
        batch = datas[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        for data in batch:

            #--------------------------#
            #           graph          # 
            #--------------------------#

            # test entities & triples
            text_entities = data["text_mentioned_entities"]
            # 1. 1hop full triples
            test_triples = None
            if config.curr_to_test == "1hop_subgraph":
                entities_selected = get_k_hop_entities(data["subgraph_before"], text_entities, k_hop)
                after_triples = get_triples_by_entities(data["subgraph_after"], entities_selected)
                test_triples, labels = get_all_possible_triples(set(after_triples), set(entities_selected), set([i for i in range(config.relation_size)]))
            # 2. added triples
            elif config.curr_to_test == "1hop_added":
                test_triples = list(set(data["subgraph_after"]) - set(data["subgraph_before"]))
                labels = [1.] * len(test_triples)
            # 3. deleted triples 
            elif config.curr_to_test == "1hop_deleted":
                test_triples = list(set(data["subgraph_before"]) - set(data["subgraph_after"]))
                labels = [0.] * len(test_triples)
            # 4. unchanged triples
            elif config.curr_to_test == "1hop_unchanged":
                test_triples = list(set(data["subgraph_before"]) & set(data["subgraph_after"]))
                labels = [1.] * len(test_triples)
            # 5. text-mentioned triples
            elif config.curr_to_test == "text_subgraph":
                test_triples, labels = get_all_possible_triples(set(data["subgraph_after"]), set(text_entities), set([i for i in range(config.relation_size)]))
            elif config.curr_to_test == "text_added":
                text_triples, _ = get_all_possible_triples(set(data["subgraph_after"]), set(text_entities), set([i for i in range(config.relation_size)]))
                added_triples = list(set(data["subgraph_after"]) - set(data["subgraph_before"]))
                test_triples    = list(set(text_triples) & set(added_triples))
                labels = [1.] * len(test_triples)
            elif config.curr_to_test == "text_deleted":
                text_triples, _ = get_all_possible_triples(set(data["subgraph_after"]), set(text_entities), set([i for i in range(config.relation_size)]))
                deleted_triples = list(set(data["subgraph_before"]) - set(data["subgraph_after"]))
                test_triples    = list(set(text_triples) & set(deleted_triples))
                labels = [0.] * len(test_triples)
            elif config.curr_to_test == "text_unchanged":
                text_triples, _ = get_all_possible_triples(set(data["subgraph_after"]), set(text_entities), set([i for i in range(config.relation_size)]))
                unchanged_triples = list(set(data["subgraph_before"]) & set(data["subgraph_after"]))
                test_triples    = list(set(text_triples) & set(unchanged_triples))
                labels = [1.] * len(test_triples)
            if not test_triples:
                continue
            triple_head_indices, triple_relation_indices, triple_tail_indices = zip(*test_triples)

            #--------------------------#
            #           adj            # 
            #--------------------------#

            # adjacent matrices
            if config.shortcut_setting == "no":
                adjacents = get_adjacents_selfloop(big_graphs[data["season"]], config.entity_size, config.relation_size)
            elif config.shortcut_setting == "unified":
                adjacents = get_adjacents_shortcut(big_graphs[data["season"]], text_entities, config.entity_size, config.relation_size)
            elif config.shortcut_setting == "labeled":
                adjacents = get_adjacents_shortcut(big_graphs[data["season"]], text_entities, config.entity_size, config.relation_size)
                label_head_indices, label_tail_indices, shortcut_labels = get_ie_label(text_entities, data["event"], rosters[data["season"]])
            else:
                print("invalid shortcut_setting:", config.shortcut_setting)
                exit()

            #--------------------------#
            #           text           # 
            #--------------------------#

            # text
            text_indices, text_length = padding_sequence(data["text"], max_length=config.text_max_length, pad_idx=config.pad_idx, get_length=True)

            # generate input and output dict
            input_dict = { 
                "entity_indices": torch.arange(config.entity_size).cuda(),
                "text_indices": torch.unsqueeze(torch.LongTensor(text_indices), dim=0).cuda(),
                "text_lengths": torch.LongTensor([text_length]).cuda(),
                "triple_head_indices": torch.LongTensor(triple_head_indices).cuda(),
                "triple_relation_indices": torch.LongTensor(triple_relation_indices).cuda(),
                "triple_tail_indices": torch.LongTensor(triple_tail_indices).cuda()
            }
            input_dict.update([("adjacent_%s" % i, adjacent.cuda()) for i, adjacent in enumerate(adjacents)])

            output_dict = {
                "score": torch.FloatTensor(labels).cuda()
            }

            if config.shortcut_setting == "labeled":
                input_dict["label_head_indices"] = torch.LongTensor(label_head_indices).cuda()
                input_dict["label_tail_indices"] = torch.LongTensor(label_tail_indices).cuda()
                output_dict["shortcut_labels"] = torch.LongTensor(shortcut_labels).cuda()

            yield input_dict, output_dict


def test():
    
    # init config
    print("init config ...")
    config = GUpdaterConfig()
    model_name = config.model_name
    # select best model
    max_f1, max_model = 0, ""
    for name in os.listdir(os.path.join(path_model_dir, model_name)):
        f1 = float(name.split("_")[-1])
        if f1 > max_f1:
            max_f1 = f1 
            max_model = name
    path_model = os.path.join(path_model_dir, model_name, max_model)

    # load data 
    print("load data ...")
    datas_train = load_NBAtransactions(path_train) 
    datas_dev   = load_NBAtransactions(path_valid)
    datas_test  = load_NBAtransactions(path_test)
    big_graphs  = gen_big_graphs(datas_train + datas_dev + datas_test)
    rosters     = {season:get_roster(big_graph) for season, big_graph in big_graphs.items()}
    print("test data size: ", len(datas_test))

    # gpu config
    print("set gpu and init model ...")
    print("load model from:", path_model)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    
    # test
    print("start testing ...")

    for test_item in config.to_test:
        print("test item:", test_item)
        config.curr_to_test = test_item
        # model
        model = GUpdater(config).cuda()
        model.load_state_dict(torch.load(path_model))
        model.eval()

        data_size = 0
        datas_test  = [data for data in datas_test if data["text"]]
        acc_num, total_num = 0, 0
        label_acc_num, label_total_num = 0, 0
        TP, FP, TN, FN = 0, 0, 0, 0
        epsilon = 1e-30

        with torch.no_grad():
            for input_dict, output_dict in data_generator(datas_test, big_graphs, rosters, config, shuffle=False, k_hop=config.k_hop):
                scores = model(input_dict)     

                if config.shortcut_setting == "labeled":
                    scores, label_scores = scores
                    label_total_num += len(label_scores)
                    label_acc_num += torch.sum(torch.argmax(label_scores, dim=-1) == output_dict["shortcut_labels"]).item()

                acc_num   += compute_acc(scores, output_dict["score"])
                total_num += len(output_dict["score"])
                TP += torch.sum(torch.sign((scores >= 0.5).float() * output_dict["score"])).item()
                FN += torch.sum(torch.sign((scores <  0.5).float() * output_dict["score"])).item()
                FP += torch.sum(torch.sign((scores >= 0.5).float() * (1 - output_dict["score"]))).item()
                TN += torch.sum(torch.sign((scores <  0.5).float() * (1 - output_dict["score"]))).item()

        accuracy  = acc_num / (total_num + epsilon)
        label_accuracy = label_acc_num / (label_total_num + epsilon)
        precision = TP / (TP + FP + epsilon)
        recall    = TP / (TP + FN + epsilon)
        f1        = 2 * precision * recall / (precision + recall + epsilon)
        
        print("accuracy: %d/%d=%.4f" % (acc_num, total_num, accuracy))
        if config.shortcut_setting == "labeled":
            print("label_accuracy: %d/%d=%.4f" % (label_acc_num, label_total_num, label_accuracy))
        if "subgraph" in test_item:
            print("precision: %.4f,\t recall: %.4f,\t f1: %.4f" % (precision, recall, f1))
        print()


    

# ====================== main =========================

def main():
    test()

if __name__ == '__main__':
    main()