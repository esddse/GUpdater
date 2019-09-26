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

            # entities & triples
            text_entities = data["text_mentioned_entities"]
            entities_selected = get_k_hop_entities(data["subgraph_before"], text_entities, k_hop)
            after_triples = get_triples_by_entities(data["subgraph_after"], entities_selected)
            
            # get all possible triples
            triples, labels = get_all_possible_triples(set(after_triples), set(entities_selected), set([i for i in range(config.relation_size)]))
            # shuffle
            triples_labels = list(zip(triples, labels))
            random.shuffle(triples_labels)
            triples, labels = zip(*triples_labels)
            triple_head_indices, triple_relation_indices, triple_tail_indices = zip(*triples)

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


def train():
    
    # init config
    print("init config ...")
    config = GUpdaterConfig()

    # create dir
    model_name = config.model_name
    path_model = os.path.join(path_model_dir, model_name)
    if not os.path.exists(path_model):
        os.mkdir(path_model)
    print("model name:", model_name)

    # load data 
    print("load data ...")
    datas_train = load_NBAtransactions(path_train) 
    datas_dev   = load_NBAtransactions(path_valid)
    datas_test  = load_NBAtransactions(path_test)
    big_graphs  = gen_big_graphs(datas_train + datas_dev + datas_test)
    rosters     = {season:get_roster(big_graph) for season, big_graph in big_graphs.items()}
    train_size  = len(datas_train)
    print("train data size: ", train_size)

    # gpu config
    print("set gpu and init model ...")
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

    # model & load
    model = GUpdater(config).cuda()

    # train
    loss_func = nn.BCELoss()
    label_loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=config.patience, factor=config.decay_factor, verbose=True, min_lr=config.min_learning_rate)
    min_dev_loss = float("inf")
    max_f1 = 0
    epsilon = 1e-30
    last_path = ""
    for epoch in range(config.epoch_num):
        # train
        model.train()
        print("start epoch %d:" % epoch)
        print("======")
        train_loss = 0
        train_label_loss = 0 
        pos_avg, neg_avg = 0, 0
        acc_num, total_num = 0, 0 
        label_acc_num, label_total_num = 0, 0
        TP, FP, TN, FN = 0, 0, 0, 0
        precision, recall, f1 = 0, 0, 0
        step = 0
        time_start = time.time()
        for input_dict, output_dict in data_generator(datas_train, big_graphs, rosters, config, k_hop=config.k_hop):
            model.zero_grad()
            scores = model(input_dict)
            if config.shortcut_setting == "labeled":
                scores, label_scores = scores
                update_loss = loss_func(scores, output_dict["score"])
                label_loss = label_loss_func(label_scores, output_dict["shortcut_labels"])  
            else:
                loss = loss_func(scores, output_dict["score"])
            # metrics
            pos_avg += my_pos_avg(scores, output_dict["score"])
            neg_avg += my_neg_avg(scores, output_dict["score"])
            acc_num += compute_acc(scores, output_dict["score"])
            total_num += len(output_dict["score"])

            # label accuracy
            if config.shortcut_setting == "labeled":
                label_total_num += len(label_scores)
                label_acc_num += torch.sum(torch.argmax(label_scores, dim=-1) == output_dict["shortcut_labels"]).item()
                label_loss *= (1. - label_acc_num/label_total_num)
                loss = update_loss + label_loss

            TP += torch.sum(torch.sign((scores >= 0.5).float() * output_dict["score"])).item()
            FN += torch.sum(torch.sign((scores <  0.5).float() * output_dict["score"])).item()
            FP += torch.sum(torch.sign((scores >= 0.5).float() * (1 - output_dict["score"]))).item()
            TN += torch.sum(torch.sign((scores <  0.5).float() * (1 - output_dict["score"]))).item()
            precision = TP / (TP + FP + epsilon)
            recall    = TP / (TP + FN + epsilon)
            f1        = 2 * precision * recall / (precision + recall + epsilon)

            loss.backward()
            optimizer.step()

            train_loss += loss 
            if config.shortcut_setting == "labeled":
                train_label_loss += label_loss
            step += 1
            if step % config.report_step_num == 0:
                time_end = time.time()
                if config.shortcut_setting == "labeled":
                    print("epoch %d, step: %d,  %%%.2f/%%100, train_loss: %.4f, train_label_loss: %.4f, train_acc: %.4f, label_acc: %.4f, precision: %.4f, recall: %.4f, f1: %.4f,  time_spent: %.2f min" % 
                         (epoch, step, step/train_size*100, train_loss/step, train_label_loss/step, acc_num/total_num, label_acc_num/label_total_num, precision, recall, f1, (time_end-time_start)/60))
                else:
                    print("epoch %d, step: %d,  %%%.2f/%%100, train_loss: %.4f, train_acc: %.4f, precision: %.4f, recall: %.4f, f1: %.4f,  time_spent: %.2f min" % 
                         (epoch, step, step/train_size*100, train_loss/step, acc_num/total_num, precision, recall, f1, (time_end-time_start)/60))
                

        print("*****")
        # dev
        model.eval()
        dev_loss = 0
        dev_label_loss = 0
        step = 0
        acc_num, total_num = 0, 0 
        label_acc_num, label_total_num = 0, 0
        TP, FP, TN, FN = 0, 0, 0, 0
        precision, recall, f1 = 0, 0, 0
        with torch.no_grad():
            for input_dict, output_dict in data_generator(datas_dev, big_graphs, rosters, config, k_hop=config.k_hop):
                scores = model(input_dict)
                if config.shortcut_setting == "labeled":
                    scores, label_scores = scores
                    update_loss = loss_func(scores, output_dict["score"])
                    label_loss = label_loss_func(label_scores, output_dict["shortcut_labels"])
                else:
                    loss = loss_func(scores, output_dict["score"])
                acc_num += compute_acc(scores, output_dict["score"])

                # label accuracy
                if config.shortcut_setting == "labeled":
                    label_total_num += len(label_scores)
                    label_acc_num += torch.sum(torch.argmax(label_scores, dim=-1) == output_dict["shortcut_labels"]).item()
                    label_loss *= (1. - label_acc_num/label_total_num)
                    loss = update_loss + label_loss
                total_num += len(output_dict["score"])
                TP += torch.sum(torch.sign((scores >= 0.5).float() * output_dict["score"])).item()
                FN += torch.sum(torch.sign((scores <  0.5).float() * output_dict["score"])).item()
                FP += torch.sum(torch.sign((scores >= 0.5).float() * (1 - output_dict["score"]))).item()
                TN += torch.sum(torch.sign((scores <  0.5).float() * (1 - output_dict["score"]))).item()
                dev_loss += loss
                if config.shortcut_setting == "labeled":
                    dev_label_loss += label_loss
                step += 1
        precision = TP / (TP + FP + epsilon)
        recall    = TP / (TP + FN + epsilon)
        f1        = 2 * precision * recall / (precision + recall + epsilon)
        if config.shortcut_setting == "labeled":
            print("epoch %d dev, dev_loss: %.4f, dev_label_loss: %.4f, dev_acc: %.4f, dev_label_acc: %.4f, precision: %.4f, recall: %.4f, f1: %.4f" % 
                 (epoch, dev_loss/step, dev_label_loss/step, acc_num/total_num, label_acc_num/label_total_num, precision, recall, f1))
        else:
            print("epoch %d dev, dev_loss: %.4f, dev_acc: %.4f, precision: %.4f, recall: %.4f, f1: %.4f" % 
                 (epoch, dev_loss/step, acc_num/total_num, precision, recall, f1))

        # save
        if f1 > max_f1:
            if last_path:
                print("remove %s" % (last_path))
                os.remove(last_path)
            ckpt_name = "epoch_%d_loss_%.4f_f1_%.4f" % (epoch, dev_loss/step, f1)
            save_path = os.path.join(path_model, ckpt_name)
            last_path = save_path
            print("f1 from %.4f -> %.4f, saving model to %s" % (max_f1, f1, save_path))
            torch.save(model.state_dict(), save_path)
            max_f1 = f1
        # lr scheduler
        scheduler.step(f1)

        print()

    

# ====================== main =========================

def main():
    train()

if __name__ == '__main__':
    main()