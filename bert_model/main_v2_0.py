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

#########################################################################
# Initialize arguments
##########################################################################
gpu, config, general, params, gt_params, encoder_params, MODEL_NAME, out_dir = get_params()
logger.info('-' * 100)
logger.info('Input Argument Information')
logger.info('-' * 100)

all_args_dict = {**gpu, **config, **general, **params, **params['schedule'], **params['lr'], **params['loss'],
                 **gt_params, **encoder_params, "out_dir": out_dir, "model_name": MODEL_NAME}

for a in all_args_dict:
    if not isinstance(all_args_dict[a], dict):
        logger.info('%-28s  %s' % (a, all_args_dict[a]))

args = argparse.Namespace(**all_args_dict)

# save args
os.makedirs(out_dir, exist_ok=True)
args_file = "GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y') + "_" + "training_args.bin"
torch.save(all_args_dict, os.path.join(out_dir, args_file))
args.exp_name = args.out_dir

#########################################################################
# Read Data
##########################################################################
tokenizer = MODEL_CLASSES[args.model_encoder_type][2].from_pretrained(args.encoder_name_or_path)

train_dataset, train_sample_weights = load_and_cache_examples(args, tokenizer, mode="train")
dev_dataset, dev_sample_weights = load_and_cache_examples(args, tokenizer, mode="dev")
test_dataset, test_sample_weights = load_and_cache_examples(args, tokenizer, mode="test")

print("train_dataset: ", len(train_dataset))
print("train_sample_weights: ", len(train_sample_weights))
print("dev_dataset: ", len(dev_dataset))
print("dev_sample_weights: ", len(dev_sample_weights))
print("test_dataset: ", len(test_dataset))
print("test_sample_weights: ", len(test_sample_weights))
#########################################################################
# Initialize Model
##########################################################################
trainer = Trainer(
    args,
    train_dataset=train_dataset,
    dev_dataset=dev_dataset,
    test_dataset=test_dataset,
    train_sample_weights=train_sample_weights,
    dev_sample_weights=dev_sample_weights,
    test_sample_weights=test_sample_weights,
)

global_step, tr_loss, train_loss_all, dev_loss_all = trainer.train()
print(train_loss_all)
print(dev_loss_all)
trainer.load_model()
trainer.evaluate("dev")
trainer.evaluate("test")
