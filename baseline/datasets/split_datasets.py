# -*- coding:utf-8 -*-
import os
import random
from collections import defaultdict


def split_train(train_file, dev_ratio=0.2, to_folder=None):

    # split train into train & dev
    with open(train_file, "r", encoding="utf-8") as f:
        dict_label_name2sents = defaultdict(list)
        for i, line in enumerate(f):
            if i == 0:
                continue

            line = line.strip()
            if not line:
                continue

            id, sent, label_name = line.split(",")
            dict_label_name2sents[label_name].append(sent)

        to_train_file = os.path.join(to_folder, "train.txt")
        to_dev_file = os.path.join(to_folder, "dev.txt")
        to_test_file = os.path.join(to_folder, "test.txt")

        train_samples = []
        dev_samples = []
        for label_name, sents in dict_label_name2sents.items():
            random.shuffle(sents)

            train_sents_ = sents[int(dev_ratio * len(sents)) + 1: ]
            dev_sents_ = sents[: int(dev_ratio * len(sents)) + 1]

            train_samples.extend(
                [(w, label_name) for w in train_sents_]
            )
            dev_samples.extend(
                [(w, label_name) for w in dev_sents_]
            )

        for samps, file_path in zip([train_samples, dev_samples], [to_train_file, to_dev_file]):
            f_out = open(file_path, "w", encoding="utf-8")
            for i, samp in enumerate(samps):
                f_out.write("%d,%s,%s" % (i, samp[0], samp[1]) + "\n")


def split_train_to_5folds(train_file, to_folder, num_folds=5):
    os.makedirs(to_folder, exist_ok=True)

    for i in range(num_folds):
        to_folder_i = os.path.join(to_folder, "fold_%d" % i)
        os.makedirs(to_folder_i, exist_ok=True)
        split_train(train_file, dev_ratio=0.2, to_folder=to_folder_i)


if __name__ == '__main__':
    train_file = "/data2/nlpData/daguanfengxian/datagrand_2021_train.csv"
    to_folder = "/data2/nlpData/daguanfengxian/phase_1/splits"
    split_train_to_5folds(train_file, to_folder, num_folds=5)
