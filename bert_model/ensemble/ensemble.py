# -*- coding:utf-8 -*-
"""
    IMPORTING LIBS
"""
import numpy as np
import pandas as pd
import os
import logging
import socket
import time
import random
import glob
import argparse, json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader
from os.path import join
from tensorboardX import SummaryWriter

from tqdm import tqdm
from tqdm import trange
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
import torchsnooper
from collections import Counter

"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from args_config import get_params
from models.model_envs import MODEL_CLASSES
from dataload.data_loader_bert import load_and_cache_examples
from training.Trainer import Trainer
from dataload.data_loader_bert import get_labels

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
model_root_path = "/data2/code/DaguanFengxian/bert_model/data/outputs/"

checkpoint_list_first = [
    "train.bert_base_first_v2_ce_class_weigths",
    # "train.nezha_base_v2",
    "train.nezha_base_wwm_dice_supconloss_0.1_0.5_sace_dr_msdrop_sum_4_0.4_hongfan__v3",
    "ensemble/train.nezha_base_wwm_dice_supconloss_0.1_0.5_sace_dr_msdrop_sum_4_0.4_hongfan__v3_fold1",
    "ensemble/train.nezha_base_wwm_dice_supconloss_0.1_0.5_sace_dr_msdrop_sum_4_0.4_hongfan__v3_fold2",
    # "ensemble/train.nezha_base_wwm_dice_supconloss_0.1_0.5_sace_dr_msdrop_sum_4_0.4_hongfan__v3_fold3",
    # "ensemble/train.nezha_base_wwm_dice_supconloss_0.1_0.5_sace_dr_msdrop_sum_4_0.4_hongfan__v3_fold4",

    "train.nezha_base_wwm_dice_supconloss_0.1_0.5_sace_dr_msdrop_sum_4_0.4__v3",
    "train.nezha_base_wwm_dice_ntx_0.1_0.5_sace_dr_msdrop_sum_4_0.4__v3",
    "train.nezha_base_wwm_ce_ntx_0.1_0.5_sa_dr_msdrop_sum_4_0.4__v3",
]
# 好坏都有
checkpoint_list_two = [
    "train.albert_base_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4__v3.json",
    "ensemble/train.bert120k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold4__v3",
    "ensemble/train.bert120k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold1__v3",
    "ttrain.bert120k_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm__v3",
    "ttrain.bert120k_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fgm__v3",
    "train.bert120k_focal_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_hongfan_grulstm__v3",

    "train.bert_base_first_v2_ce_class_weigths",
    "train.nezha_base_wwm_dice_supconloss_0.1_0.5_sace_dr_msdrop_sum_4_0.4_hongfan__v3",
    "ensemble/train.nezha_base_wwm_dice_supconloss_0.1_0.5_sace_dr_msdrop_sum_4_0.4_hongfan__v3_fold1",
    "ensemble/train.nezha_base_wwm_dice_supconloss_0.1_0.5_sace_dr_msdrop_sum_4_0.4_hongfan__v3_fold2",
    "train.nezha_base_wwm_dice_supconloss_0.1_0.5_sace_dr_msdrop_sum_4_0.4__v3",
    "train.nezha_base_wwm_dice_ntx_0.1_0.5_sace_dr_msdrop_sum_4_0.4__v3",
    "train.nezha_base_wwm_ce_ntx_0.1_0.5_sa_dr_msdrop_sum_4_0.4__v3",
]

checkpoint_list_three = [
    "ensemble/train.bert120k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold4__v3",
    "ensemble/train.bert120k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold1__v3",
    "ensemble/train.bert120k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold2__v3",
    "ensemble/train.bert120k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold3__v3",
    "ensemble/train.bert120k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold5__v3",
    "ttrain.bert120k_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm__v3",
    "ttrain.bert120k_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fgm__v3",
    "ensemble/train.nezha_base_wwm_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold1__v3",
    "ensemble/train.nezha_base_wwm_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold2__v3",
    "ensemble/train.nezha_base_wwm_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold3__v3",
    "ensemble/train.nezha_base_wwm_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold4__v3",
    "train.nezha_base_wwm_dice_supconloss_0.1_0.5_sace_dr_msdrop_sum_4_0.4_hongfan__v3",
]

checkpoint_list_four = [
    "new_ensemble/train.bert150k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold0_lr2e-5__v3",
    "train.bert_base_first_v2_ce_class_weigths",
    "train.nezha_base_wwm_ce_ntx_0.1_0.5_sa_dr_msdrop_sum_4_0.4__v3",
    "ensemble/train.nezha_base_wwm_dice_supconloss_0.1_0.5_sace_dr_msdrop_sum_4_0.4_hongfan__v3_fold1",
    "ensemble/train.nezha_base_wwm_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold4__v3",
    "ensemble/train.bert120k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold4__v3",
    "ttrain.bert120k_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fgm__v3",
    "new_ensemble/train.bert150k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold0_fgm__v3",
]

checkpoint_list_five = [
    "new_ensemble/train.bert150k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold0_lr2e-5__v3",
    "new_ensemble/train.bert150k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold0_fgm__v3",
    "new_ensemble/train.bert120k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold4__v3",
    "final_ensemble/train.bert300_dice_supcon_lr1e-5_pgd__fold0__v3",
    "final_ensemble/train.bert300_dice_supcon_nolstmgru_lr2e-5_pgd__fold0__v3",
    "final_ensemble/train.newnezha_dice_supcon_lr1e-5_nomulti_fgm__fold0__v3",
    "final_ensemble/train.newnezha_dice_supcon_lr5e-5_nomulti_pgd__fold0__v3",
    "final_ensemble/train.newnezha_dice_supcon_lr7e-5__fold0__v3",
]

checkpoint_list = checkpoint_list_five


def full_args(args):
    if not hasattr(args, 'use_ms_dropout'):
        args.use_ms_dropout = False
    if not hasattr(args, 'use_hongfan'):
        args.use_hongfan = False
    if not hasattr(args, 'contrastive_loss'):
        args.contrastive_loss = None
    if not hasattr(args, 'use_focal_loss'):
        args.use_focal_loss = False
    if not hasattr(args, 'use_freelb'):
        args.use_freelb = False
    if not hasattr(args, 'use_multi_task'):
        args.use_multi_task = False
    if not hasattr(args, 'use_gru'):
        args.use_gru = False
    if not hasattr(args, 'use_swa'):
        args.use_swa = False


results_label_list = []
results_logits_list = []
preds_level_2_prob_list = []
avg_weight = []

for i, checkpoint_path in enumerate(checkpoint_list):
    logger.info("********** start:" + checkpoint_path + "**********")
    checkpoint_path = model_root_path + checkpoint_path
    args = torch.load(os.path.join(checkpoint_path + '/training_args_bin'))
    full_args(args)

    if "bert120k" in checkpoint_path:
        avg_weight.append(1)
    elif "nezha_base_wwm" in checkpoint_path:
        avg_weight.append(0.8)
    elif "bert_base" in checkpoint_path:
        avg_weight.append(0.7)
    elif "albert" in checkpoint_path:
        avg_weight.append(0.6)

    tokenizer = MODEL_CLASSES[args.model_encoder_type][2].from_pretrained(args.encoder_name_or_path)
    test_dataset, test_sample_weights = load_and_cache_examples(args, tokenizer, mode="test")

    trainer = Trainer(
        args,
        test_dataset=test_dataset,
        test_sample_weights=test_sample_weights,
    )

    trainer.load_model()
    list_preds_level_2, logits_level_2, preds_level_2_prob, _, _ = trainer.ensemble_test("test")  # [6004,35]

    results_label_list += [list_preds_level_2]  # [check_i,idx_sample]
    results_logits_list.append(logits_level_2)  # [check_i,idx_sample]
    preds_level_2_prob_list.append(preds_level_2_prob)  # [check_i,idx_sample,35]
    logger.info("********** end:" + checkpoint_path + "**********")

true_labels = get_labels("/data2/code/DaguanFengxian/bert_model/data/labels_level_2.txt")
"""
    vote
"""


def prob_avg_rank_in_list(prob, prob_list_np):  # 求一个数在二维数组每行的排名，然后求均值
    rank_list = []

    for i, element in enumerate(prob_list_np):
        rank = 0
        for p in element:
            if prob[i] < p:  # 概率大的放前面
                rank += 1
        rank_list.append(rank)

    return np.array(rank_list).mean()


vote_flag = 2  # 0 相同取靠前 1 相同取概率最大 2 相同取平均排名最高的
###### 是否使用相同投票 使用概率加和
vote_label_idx = []
len_test = len(results_label_list[0])
same_num = 0
for i in range(len_test):
    pred_list_i = []
    prob_list_i = []  # 每个模型 预测本条test的概率 [model_num, 35]
    for j in range(len(checkpoint_list)):
        pred_list_i.append(results_label_list[j][i])
        prob_list_i.append(preds_level_2_prob_list[j][i])

    if vote_flag == 1:
        ####### 票数相同选概率和最大的
        most_ = Counter(pred_list_i).most_common(35)
        max_vote_num = most_[0][1]
        most_ = [m for m in most_ if m[1] != 1]  # 剔除1票的相同者
        most_ = [m for m in most_ if m[1] == max_vote_num]  # 只选择等于投票最大值的
        if len(most_) == 0:  # 如果全是1票
            vote_label_idx.append(Counter(pred_list_i).most_common(1)[0][0])
        elif len(most_) == 1:
            vote_label_idx.append(most_[0][0])
        else:
            prob_list_np = np.array(prob_list_i)
            select_prob = -1
            select_m = -1
            same_num += 1
            for m, num in most_:
                # 拿概率第m列（所有模型对第m列的概率）求和
                prob_m = prob_list_np[:, m].sum()
                if select_prob < prob_m:
                    select_prob = prob_m
                    select_m = m
            vote_label_idx.append(select_m)
    elif vote_flag == 2:
        ####### 票数相同选平均rank最大的
        most_ = Counter(pred_list_i).most_common(35)
        max_vote_num = most_[0][1]
        most_ = [m for m in most_ if m[1] != 1]  # 剔除1票的相同者
        most_ = [m for m in most_ if m[1] == max_vote_num]  # 只选择等于投票最大值的
        if len(most_) == 0:  # 如果全是1票
            vote_label_idx.append(Counter(pred_list_i).most_common(1)[0][0])
        elif len(most_) == 1:
            vote_label_idx.append(most_[0][0])
        else:
            prob_list_np = np.array(prob_list_i)
            select_rank = -1
            select_m = -1
            same_num += 1
            for m, num in most_:
                # 拿概率第m列（所有模型对第m列的概率）求和
                prob_m = prob_list_np[:, m]
                prob_m_avgrank = prob_avg_rank_in_list(prob_m, prob_list_np)
                if select_rank > prob_m_avgrank:  # 选择排名小的 靠前的
                    select_rank = prob_m_avgrank
                    select_m = m
            vote_label_idx.append(select_m)
    else:
        vote_label_idx.append(Counter(pred_list_i).most_common(1)[0][0])  # 相同的取第一个

print("相同的投票样本有多少：", same_num)
"""
   label rank排序选择
"""

"""
    argmax
"""
argmax_label_idx = []
for i in range(len_test):
    pred_list_i_logits = []
    pred_list_i_labels = []
    for j in range(len(checkpoint_list)):
        d = results_logits_list[j][i]
        pred_list_i_logits.append(d[0])
        pred_list_i_labels.append(d[1])
    idx = np.argmax(pred_list_i_logits)
    argmax_label_idx.append(pred_list_i_labels[idx])

"""
    算数平均
"""
avg_flag = False
###### 是否使用相同投票 使用概率加和
avg_label_idx = []
for i in range(len_test):
    pred_list_i_prob = []
    pred_list_i_labels = []

    i_all_pred = []
    for j in range(len(checkpoint_list)):
        if avg_flag:
            p = preds_level_2_prob_list[j][i] * avg_weight[j]
        else:
            p = preds_level_2_prob_list[j][i]
        i_all_pred.append(p)

    i_all_pred = np.array(i_all_pred)
    i_all_pred_sum = np.sum(i_all_pred, axis=0) / len(checkpoint_list)
    idx = np.argmax(i_all_pred_sum)
    avg_label_idx.append(idx)

"""
    save
"""
index = np.arange(0, len_test)
print(index[avg_label_idx != argmax_label_idx])  # 找到两个数组不相等元素的下标位置
print(index[avg_label_idx == argmax_label_idx])  # 找到两个数组相等元素的下标位置
b = np.array(avg_label_idx) == np.array(argmax_label_idx)
print(np.array(avg_label_idx) == np.array(argmax_label_idx))
print(avg_label_idx == argmax_label_idx)

f_out = open(os.path.join("/data2/code/DaguanFengxian/bert_model/data/outputs/ensemble_vote_four_rank.csv"), "w",
             encoding="utf-8")
f_out.write("id,label" + "\n")
for i, pred_label_id in enumerate(vote_label_idx):
    pred_label_name_level_2 = true_labels[pred_label_id]
    f_out.write("%s,%s" % (str(i), str(pred_label_name_level_2)) + "\n")

f_out = open(os.path.join("/data2/code/DaguanFengxian/bert_model/data/outputs/ensemble_argmax_four.csv"), "w",
             encoding="utf-8")
f_out.write("id,label" + "\n")
for i, pred_label_id in enumerate(argmax_label_idx):
    pred_label_name_level_2 = true_labels[pred_label_id]
    f_out.write("%s,%s" % (str(i), str(pred_label_name_level_2)) + "\n")

f_out = open(os.path.join("/data2/code/DaguanFengxian/bert_model/data/outputs/ensemble_avg_four.csv"), "w",
             encoding="utf-8")
f_out.write("id,label" + "\n")
for i, pred_label_id in enumerate(avg_label_idx):
    pred_label_name_level_2 = true_labels[pred_label_id]
    f_out.write("%s,%s" % (str(i), str(pred_label_name_level_2)) + "\n")
print("finish!")
