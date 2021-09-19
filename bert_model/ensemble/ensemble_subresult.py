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

model_root_path = "/data2/code/DaguanFengxian/bert_model/data/ensemble_data/vote_csv/"

input_files_first = [
    model_root_path + "ensemble_vote_three_rank.csv",
    model_root_path + "ensemble_vote_three_prob.csv",
    model_root_path + "ensemble_avg_three.csv",
    model_root_path + "ensemble_argmax_three.csv",
    model_root_path + "ensemble_vote.csv",
]
input_files_two = [
    model_root_path + "09.20_11_8m_fanzhuan.csv",
    model_root_path + "09.20_12_16m.csv",
    model_root_path + "0917_eight.csv",
    model_root_path + "0917_six.csv",
    model_root_path + "0917ensemble_vote_four_rank.csv",
    model_root_path + "0918_nine.csv",
    model_root_path + "newBert-589.csv",
]

input_files_three = [
    model_root_path + "09.20_11_8m_fanzhuan.csv",
    model_root_path + "09.20_12_16m.csv",
    model_root_path + "0917_eight.csv",
    model_root_path + "0917_six.csv",
    model_root_path + "0917ensemble_vote_four_rank.csv",
    model_root_path + "0918_nine.csv",
    model_root_path + "newBert-589.csv",
]

input_file = input_files_two

all_res = []
hashtable = dict()

for file in input_file:
    df = pd.read_csv(file, skiprows=0, index_col=0)
    print(df.head())
    for index, row in df.iterrows():
        if index in hashtable:
            hashtable[index] = hashtable[index] + "," + str(row.label)
        else:
            hashtable[index] = str(row.label)
    all_res.append(df)

label_vote = []
diff_num = 0
for key, value in hashtable.items():
    value_list = value.split(",")
    if len(set(value_list)) != 1:  # 1195
        diff_num += 1
    most_ = Counter(value_list).most_common(1)
    label_vote.append(most_[0][0])

print("需要越策的样本数：", diff_num)

f_out = open(os.path.join("/data2/code/DaguanFengxian/bert_model/data/ensemble_data/subresult_final111111.csv"), "w",
             encoding="utf-8")
f_out.write("id,label" + "\n")
for i, lable in enumerate(label_vote):
    pred_label_name_level_2 = lable
    f_out.write("%s,%s" % (str(i), str(pred_label_name_level_2)) + "\n")
print("finish!")
