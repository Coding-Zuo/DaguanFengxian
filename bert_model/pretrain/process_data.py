# coding:utf-8

import os
import logging
import warnings
from argparse import ArgumentParser

import pandas as pd

from label2id import label2id


logging.basicConfig()
logger = logging.getLogger('')
logger.setLevel(logging.INFO)


def cut_text(text, args):
    char = [i for i in text.split(' ')]
    length = len(char)
    if length > args.max_length:
        new_char = char[:args.max_length]
        new_text = ''
        for i in new_char:
            new_text += i + ' '
        new_text = new_text.strip()
        return new_text
    else:
        return text.strip()


def process_text(args):
    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)

    train_text = train['text'].tolist()
    test_text = test['text'].tolist()
    pretrain_text = train_text + test_text

    label = train['label'].tolist()

    pretrain_sentence, train_sentence, train_sentence1, test_sentence = [], [], [], []
    for i in pretrain_text:
        pretrain_sentence.append(i.strip())

    pretrain_sentence = list(set(pretrain_sentence))

    logger.info(f'total pretrain data : {len(pretrain_sentence)}.')

    for i in train_text:
        train_sentence.append(cut_text(i, args))

    for i in range(len(train_sentence)):
        tgt_level1, tgt_level2 = label[i].split('-')
        tgt = label2id[label[i]]
        line = train_sentence[i] + '\t' + str(tgt) + '\t' + str(int(tgt_level1)-1)
        train_sentence1.append(line)

    logger.info(f'total train data : {len(train_sentence)}.')

    for i in test_text:
        test_sentence.append(cut_text(i, args))

    logger.info(f'total test data : {len(test_sentence)}.')

    return pretrain_sentence, train_sentence1, test_sentence


def write(text_list, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for i in text_list:
            f.writelines(i + '\n')

    logger.info(f'process data has been written to {out_path}.')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        '--max_length',
        type=int,
        default=128
    )

    parser.add_argument(
        '--train_path',
        type=str,
        default=''
    )

    parser.add_argument(
        '--test_path',
        type=str,
        default=''
    )

    parser.add_argument(
        '--out_path',
        type=str,
        default=''
    )

    warnings.filterwarnings('ignore')
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    out_pretrain_path = os.path.join(args.out_path, 'pretrain.txt')
    out_train_path = os.path.join(args.out_path, 'train.txt')
    out_test_path = os.path.join(args.out_path, 'test.txt')

    pretrain, train, test = process_text(args)

    write(pretrain, out_pretrain_path)
    write(train, out_train_path)
    write(test, out_test_path)

    logger.info(f'data processing completed .')
