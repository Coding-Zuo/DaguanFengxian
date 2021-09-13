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

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
model_root_path = "/data2/code/DaguanFengxian/bert_model/data/outputs/"

bert120k_6folds = [
    # "ttrain.bert120k_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm__v3",
    "ensemble/train.bert120k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold1__v3",
    "ensemble/train.bert120k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold2__v3",
    "ensemble/train.bert120k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold3__v3",
    "ensemble/train.bert120k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold4__v3",
    "ensemble/train.bert120k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold5__v3",
]

bert_wwm_6folds = [
    "ensemble/train.bert_base_wwm_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold0__v3",
    "ensemble/train.bert_base_wwm_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold1__v3",
    "ensemble/train.bert_base_wwm_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold2__v3",
    "ensemble/train.bert_base_wwm_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold3__v3",
    "ensemble/train.bert_base_wwm_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold4__v3",
    "ensemble/train.bert_base_wwm_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold5__v3",
]

nezha_6folds = [
    "ensemble/train.nezha_base_wwm_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold0_v3",
    "ensemble/train.nezha_base_wwm_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold1__v3",
    "ensemble/train.nezha_base_wwm_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold2__v3",
    "ensemble/train.nezha_base_wwm_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold3__v3",
    "ensemble/train.nezha_base_wwm_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold4__v3",
    "ensemble/train.nezha_base_wwm_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold5__v3",
]

nezha_5folds = [
    "train.nezha_base_wwm_dice_supconloss_0.1_0.5_sace_dr_msdrop_sum_4_0.4_hongfan__v3",
    "ensemble/train.nezha_base_wwm_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_hongfan__v3_fold1",
    "ensemble/train.nezha_base_wwm_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_hongfan__v3_fold2",
    "ensemble/train.nezha_base_wwm_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_hongfan__v3_fold3",
    "ensemble/train.nezha_base_wwm_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_hongfan__v3_fold4",
]

all_model = [bert120k_6folds, nezha_6folds]

# 二级分类器的trian_x, test
# dataset_blend_train = np.zeros((11188, len(all_model)), dtype=np.int)
# dataset_blend_test  = np.zeros((6004, len(all_model)), dtype=np.int)

for model_kfold in all_model:

    prob_kfold_list = []
    for checkpoint in model_kfold:
        logger.info("********** start:" + checkpoint + "**********")
        checkpoint_path = model_root_path + checkpoint
        args = torch.load(os.path.join(checkpoint_path + '/training_args_bin'))

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

        tokenizer = MODEL_CLASSES[args.model_encoder_type][2].from_pretrained(args.encoder_name_or_path)
        test_dataset, test_sample_weights = load_and_cache_examples(args, tokenizer, mode="test")  # 6004
        trainer = Trainer(args, test_dataset=test_dataset, test_sample_weights=test_sample_weights, )
        trainer.load_model()
        list_preds_level_2, logits_level_2, preds_level_2_prob = trainer.ensemble_test()
        prob_kfold_list.append(preds_level_2_prob)
