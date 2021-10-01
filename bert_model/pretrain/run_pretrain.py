# coding:utf-8

import os
import pickle
import random
import warnings
import logging
from argparse import ArgumentParser
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from transformers import (
    BertTokenizer,
    TrainingArguments,
    Trainer,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForWholeWordMask as DGDataCollator, PreTrainedTokenizer
)


warnings.filterwarnings('ignore')

logging.basicConfig()
logger = logging.getLogger('第五届达观杯')
logger.setLevel(logging.INFO)


def save_pickle(dic, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(dic, f)


def load_pickle(load_path):
    with open(load_path, 'rb') as f:
        message_dict = pickle.load(f)
    return message_dict


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args):

        with open(args.pretrain_data_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=args.seq_length)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


def main():
    parser = ArgumentParser()

    parser.add_argument(
        '--debug',
        type=bool,
        default=False
    )

    parser.add_argument(
        '--pretrain_data_path',
        type=str,
        default=''
    )

    parser.add_argument(
        '--pretrain_model_path',
        type=str,
        default=f''
    )

    parser.add_argument(
        '--data_cache',
        type=str,
        default=f''
    )

    parser.add_argument(
        '--vocab_path',
        type=str,
        default=f''
    )

    parser.add_argument(
        '--save_path',
        type=str,
        default=''
    )

    parser.add_argument(
        '--record_save_path',
        type=str,
        default=''
    )

    parser.add_argument(
        '--mlm_probability',
        type=float,
        default=0.15
    )

    parser.add_argument(
        '--num_train_epochs',
        type=int,
        default=300
    )

    parser.add_argument(
        '--seq_length',
        type=int,
        default=128
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=6e-5
    )

    parser.add_argument(
        '--save_steps',
        type=int,
        default=10000
    )

    parser.add_argument(
        '--ckpt_save_limit',
        type=int,
        default=5
    )

    parser.add_argument(
        '--logging_steps',
        type=int,
        default=2000
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42
    )

    parser.add_argument(
        '--fp16',
        type=str,
        default=True
    )

    parser.add_argument(
        '--fp16_backend',
        type=str,
        default='amp'
    )

    warnings.filterwarnings('ignore')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.record_save_path), exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.vocab_path)
    model_config = BertConfig.from_pretrained(args.pretrain_model_path)

    data_collator = DGDataCollator(mlm=True,
                                   tokenizer=tokenizer,
                                   mlm_probability=args.mlm_probability)

    if args.pretrain_model_path is not None:
        model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path=args.pretrain_model_path,
                                                config=model_config)
    else:
        model = BertForMaskedLM(config=model_config)
    model.resize_token_embeddings(tokenizer.vocab_size)

    dataset = LineByLineTextDataset(tokenizer, args)

    training_args = TrainingArguments(
        seed=args.seed,
        fp16=args.fp16,
        fp16_backend=args.fp16_backend,
        save_steps=args.save_steps,
        prediction_loss_only=True,
        logging_steps=args.logging_steps,
        output_dir=args.record_save_path,
        learning_rate=args.learning_rate,
        save_total_limit=args.ckpt_save_limit,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(args.save_path)
    tokenizer.save_pretrained(args.save_path)


if __name__ == '__main__':
    main()
