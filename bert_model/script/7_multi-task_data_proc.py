# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

data_dir = "/data2/code/DaguanFengxian/bert_model/data/splits/fold_0_nezha_base_vocab"
data_dev = "/dev.txt"
data_train = "/train.txt"

with open(data_dir + data_train, 'r', encoding='utf-8') as f:
    train_data = []

    for i, line in enumerate(f):
        if i == 0: continue

        idx = line.strip().split("\t")[0]
        sent_ = line.strip().split("\t")[1]
        label_2 = line.strip().split("\t")[2]
        label_1 = label_2.split("-")[0]
        train_data.append([idx, sent_, label_2, label_1])

with open(data_dir + data_dev, 'r', encoding='utf-8') as f:
    dev_data = []

    for i, line in enumerate(f):
        if i == 0: continue

        idx = line.strip().split("\t")[0]
        sent_ = line.strip().split("\t")[1]
        label_2 = line.strip().split("\t")[2]
        label_1 = label_2.split("-")[0]
        dev_data.append([idx, sent_, label_2, label_1])

f_out = open(data_dir + data_train, 'w', encoding='utf-8')
for i, samp in enumerate(train_data):
    f_out.write("%d,%s,%s,%s" % (int(samp[0]), samp[1], samp[2], samp[3]) + "\n")

f_out = open(data_dir + data_dev, 'w', encoding='utf-8')
for i, samp in enumerate(dev_data):
    f_out.write("%d,%s,%s,%s" % (int(samp[0]), samp[1], samp[2], samp[3]) + "\n")
