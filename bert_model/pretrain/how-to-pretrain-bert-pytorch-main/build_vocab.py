# coding:utf-8

import os
import random
import logging
import warnings
import numpy as np
from tqdm.auto import tqdm
from argparse import ArgumentParser
from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer

import torch
from torch import nn


logging.basicConfig()
logger = logging.getLogger('build vocab')
logger.setLevel(logging.INFO)


def set_seed(manual_seed):
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manual_seed)


def train_tokenizer(args):
    tokenizer = BertWordPieceTokenizer(
        clean_text=False,
        handle_chinese_chars=True,
        strip_accents=False,
        lowercase=False
    )
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    for i in range(100):
        special_tokens.append(f"[unused{i}]")

    tokenizer.train(
        files=[args.file_path],
        vocab_size=args.vocab_size,
        min_frequency=1,
        special_tokens=special_tokens,
        limit_alphabet=args.vocab_size,
        wordpieces_prefix="##"
    )
    os.makedirs(args.out_path, exist_ok=True)
    tokenizer.save_model(args.out_path)
    tokenizer = BertTokenizer.from_pretrained(args.out_path,
                                              do_lower_case=False,
                                              strip_accents=False)
    tokenizer.save_pretrained(args.out_path)
    logger.info(f'save tokenizer, with vocab_size: {tokenizer.vocab_size}')


if __name__ == '__main__':

    model_type = 'bert'

    parser = ArgumentParser()

    parser.add_argument(
        '--seed',
        type=int,
        default=42
    )

    parser.add_argument(
        '--vocab_size',
        type=int,
        default=21128
    )

    parser.add_argument(
        '--file_path',
        type=str,
        default=''
    )

    parser.add_argument(
        '--out_path',
        type=str,
        default=f''
    )

    warnings.filterwarnings('ignore')
    args = parser.parse_args()

    set_seed(args.seed)
    train_tokenizer(args)

    logger.info(f'vocab creation completed .')
