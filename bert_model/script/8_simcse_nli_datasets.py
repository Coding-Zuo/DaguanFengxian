# -*- coding:utf-8 -*-
import argparse
import logging
import os
import pandas as pd
#### Just some code to print debug information to stdout
import random
import sys

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO, )

label_list_level_1 = [
    label.strip() for label in open("/data2/code/DaguanFengxian/bert_model/data/labels_level_1.txt")
]

label_list_level_2 = [
    label.strip() for label in open("/data2/code/DaguanFengxian/bert_model/data/labels_level_2.txt")
]


def get_label_name2sents(train_data_path):
    dict_label_name2_sents = {}
    with open(train_data_path, 'r', encoding='utf8') as fin:
        for i, row in enumerate(fin):
            row = row.strip()
            if not row:
                continue
            row = row.split(",")
            sent = row[1].strip()
            label_name = row[2].strip()

            if label_name not in dict_label_name2_sents:
                dict_label_name2_sents[label_name] = set()
            dict_label_name2_sents[label_name].add(sent)
    dict_label_name2_sents_new = {}
    for label_name, sents in dict_label_name2_sents.items():
        dict_label_name2_sents_new[label_name] = list(sents)
    return dict_label_name2_sents_new


parser = argparse.ArgumentParser()

"""

python src/SimCSE/daguan_task/prepare_nli_datasets.py 
--dataset_path datasets/phase_1/splits/fold_0 
--nli_dataset_path datasets/phase_1/splits/fold_0_nli 
--sampling_times 100000
"""
parser.add_argument("--dataset_path", default="/data2/code/DaguanFengxian/bert_model/data/splits/fold_0", type=str)
parser.add_argument("--nli_dataset_path", default="/data2/code/DaguanFengxian/bert_model/data/splits/fold_0_nli",
                    type=str)
parser.add_argument("--sampling_times", default=100000, type=int)
args = parser.parse_args()

logging.info("Read in train dataset")
train_data_path = os.path.join(args.dataset_path, "train.txt")
dict_label_name2sents_train = get_label_name2sents(train_data_path)

logging.info("Generate nli train dataset with hard negatives")
train_samples = []
for i in range(args.sampling_times):
    label_0, label_1 = random.sample(label_list_level_2, 2)
    anchor, pos = random.sample(list(dict_label_name2sents_train[label_0]), 2)
    neg = random.choice(list(dict_label_name2sents_train[label_1]))

    train_samples.append({
        "sent0": anchor,
        "sent1": pos,
        "hard_neg": neg
    })

df_train_sample = pd.DataFrame(train_samples)
os.makedirs(args.nli_dataset_path, exist_ok=True)
df_train_sample.to_csv(
    os.path.join(args.nli_dataset_path, "nli_for_simcse.csv"),
    index=False,
    sep="\t"
)
