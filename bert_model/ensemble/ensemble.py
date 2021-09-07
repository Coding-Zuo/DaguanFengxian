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

"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from args_config import get_params
from models.model_envs import MODEL_CLASSES
from dataload.data_loader_bert import load_and_cache_examples
from training.Trainer import Trainer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
model_root_path = "/data2/code/DaguanFengxian/bert_model/data/outputs/"

checkpoint_list = [
    "train.nezha_base_wwm_dice_supconloss_0.1_0.5_sace_dr_msdrop_sum_4_0.4_hongfan__v3",
    "ensemble/train.nezha_base_wwm_dice_supconloss_0.1_0.5_sace_dr_msdrop_sum_4_0.4_hongfan__v3_fold1",
    "ensemble/train.nezha_base_wwm_dice_supconloss_0.1_0.5_sace_dr_msdrop_sum_4_0.4_hongfan__v3_fold2",
    # "ensemble/train.nezha_base_wwm_dice_supconloss_0.1_0.5_sace_dr_msdrop_sum_4_0.4_hongfan__v3_fold3",
    # "ensemble/train.nezha_base_wwm_dice_supconloss_0.1_0.5_sace_dr_msdrop_sum_4_0.4_hongfan__v3_fold4",
    "train.nezha_base_wwm_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4__v3",
    "train.nezha_base_wwm_dice_ntx_0.1_0.5_sace_dr_msdrop_sum_4_0.4__v3",
    "train.nezha_base_wwm_ce_ntx_0.1_0.5_sa_dr_msdrop_sum_4_0.4__v3",
]

results_label_list = []
results_logits_list = []

for checkpoint_path in checkpoint_list:
    logger.info("********** start:" + checkpoint_path + "**********")
    checkpoint_path = model_root_path + checkpoint_path
    args = torch.load(os.path.join(checkpoint_path + '/training_args_bin'))

    tokenizer = MODEL_CLASSES[args.model_encoder_type][2].from_pretrained(args.encoder_name_or_path)
    test_dataset, test_sample_weights = load_and_cache_examples(args, tokenizer, mode="test")

    trainer = Trainer(
        args,
        test_dataset=test_dataset,
        test_sample_weights=test_sample_weights,
    )

    trainer.load_model()
    list_preds_level_2, logits_level_2 = trainer.ensemble_test()

    results_label_list += [list_preds_level_2]
    results_logits_list += [logits_level_2]
    logger.info("********** end:" + checkpoint_path + "**********")

np.argmax(results_label_list, axis=0)
