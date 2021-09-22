# -*- coding:utf-8 -*-
import json
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import sys

sys.path.append("./")
"""
虽然无法还原句子
但频率估计可以还原一部分词
两个频率高的文本，在同一种语境下出现的概率更大
从语义相关性角度来说，可能会有一些语义相关性

改用明文后就可以随便用预训练语言模型了
"""
from models.model_utils import get_embedding_matrix_and_vocab


def get_vocab_mapping(split_num, model_type):
    vocab_path_bert_base = "/home/zuoyuhui/DataGame/haihuai_RC/chinese-bert-wwm-ext/vocab.txt"
    vocab_path_albert_xxlarge = "/data2/pre-model/albert/albert_xxlarge_chinese/vocab.txt"
    vocab_path_bert_large = "/data2/pre-model/bert/bert_large_chinese/vocab.txt"
    vocab_path_nezha_base = "/data2/pre-model/nezha/NEZHA-Base-WWM/vocab.txt"
    vocab_path_nezha_large = "/data2/pre-model/nezha/nezha-large-www/vocab.txt"
    vocab_path_bert120k = "/data2/code/DaguanFengxian/pretrain_weight/steps_120k/vocab.txt"
    vocab_path_bert150k = "/data2/code/DaguanFengxian/pretrain_weight/steps_150k/vocab.txt"
    vocab_path_macbert = "/data2/pre-model/macbert/hfl_chinese-macbert-base/vocab.txt"

    if model_type == "albert_xxlarge":
        vocab_path = vocab_path_albert_xxlarge
    elif model_type == "bert":
        vocab_path = vocab_path_bert_base
    elif model_type == "bert_large":
        vocab_path = vocab_path_bert_large
    elif model_type == "nezha_base":
        vocab_path = vocab_path_nezha_base
    elif model_type == "nezha_large":
        vocab_path = vocab_path_nezha_large
    elif model_type == "bert_120k":
        vocab_path = vocab_path_bert120k
    elif model_type == "bert_150k":
        vocab_path = vocab_path_bert150k
    elif model_type == "macbert":
        vocab_path = vocab_path_macbert
    else:
        print("名字错误")
        exit(1)

    # 先得到本次比赛的词频排序
    # w2v:先遍历一次，得到一个vocab list 和 向量的list
    w2v_file = "/data2/nlpData/daguanfengxian/word2vec/dim_768_sg_0_hs_1_epochs_30/w2v.vectors"
    daguan_freq_rank2vocab, _ = get_embedding_matrix_and_vocab(w2v_file, include_special_tokens=False)

    daguan_freq_rank2vocab.pop(daguan_freq_rank2vocab.index("，"))
    daguan_freq_rank2vocab.pop(daguan_freq_rank2vocab.index("。"))
    daguan_freq_rank2vocab.pop(daguan_freq_rank2vocab.index("！"))
    daguan_freq_rank2vocab.pop(daguan_freq_rank2vocab.index("？"))

    # 数据集中出现的词汇
    vocab_in_tasks = defaultdict(int)
    df_train = pd.read_csv("/data2/code/DaguanFengxian/bert_model/data/splits_quchong/fold_" + split_num + "/train.txt",
                           header=None)
    df_train.columns = ["id", "text", "label", "guid"]
    df_val = pd.read_csv("/data2/code/DaguanFengxian/bert_model/data/splits_quchong/fold_" + split_num + "/dev.txt",
                         header=None)
    df_val.columns = ["id", "text", "label", "guid"]
    df_test = pd.read_csv("/data2/code/DaguanFengxian/bert_model/data/splits6/test.txt", header=None)
    df_test.columns = ["id", "text", ]

    for df_ in [df_train, df_val, df_test]:
        for text in df_['text']:
            for char in text.split(" "):
                vocab_in_tasks[char] += 1

    print(vocab_in_tasks)
    print(len(vocab_in_tasks))

    # for v in vocab_in_tasks:
    #     print(v, daguan_freq_rank2vocab.index(v))

    # 得到一个中文BERT词汇频率的排序
    # counts = json.load(open('/data2/code/DaguanFengxian/bert_model/data/vocab_freq/dict_vocab2freq_0808.json'))
    counts = json.load(open('/data2/code/DaguanFengxian/bert_model/data/vocab_freq/dict_vocab2freq_0819.json'))
    # del counts["[CLS]"]
    # del counts['[SEP]']
    # del counts["[UNK]"]
    # del counts["[PAD]"]
    # del counts["[MASK]"]

    del counts["，"]
    del counts["。"]
    del counts["！"]
    del counts["？"]

    token_dict = {}
    with open(vocab_path) as reader:
        for line in reader:
            token = line.split()
            token = token[0] if token else line.strip()
            token_dict[token] = len(token_dict)

    del token_dict["[CLS]"]
    del token_dict["[SEP]"]
    del token_dict["[UNK]"]
    del token_dict["[PAD]"]
    del token_dict["[MASK]"]

    del token_dict["，"]
    del token_dict["。"]
    del token_dict["！"]
    del token_dict["？"]

    # 得到bert token 在开源语料库中的对应词频数
    list_bert_vocab2freqs = [
        (i, counts.get(i, 0)) for i, j in sorted(token_dict.items(), key=lambda s: s[1])
    ]
    print(list_bert_vocab2freqs)

    list_bert_vocab2freqs = sorted(list_bert_vocab2freqs, key=lambda x: x[1], reverse=True)
    list_bert_vocab2freq_rank = [w[0] for w in list_bert_vocab2freqs]
    print(list_bert_vocab2freq_rank)

    #
    dict_daguan_vocab2bert_vocab = {}
    count_unused = 0
    for v in vocab_in_tasks:  # 遍历任务中出现的词
        if v in daguan_freq_rank2vocab:
            v_freq_rank_in_daguan = daguan_freq_rank2vocab.index(v)

            if v_freq_rank_in_daguan < len(list_bert_vocab2freq_rank):
                v_in_bert = list_bert_vocab2freq_rank[v_freq_rank_in_daguan]
            else:
                v_in_bert = "[unused%d]" % (count_unused + 1)
                count_unused += 1
                print(v_in_bert, v)
            dict_daguan_vocab2bert_vocab[v] = v_in_bert
        else:
            print("not included in daguan_freq_rank2vocab: ", v)

    # 因为数据中没把这些标点脱敏
    dict_daguan_vocab2bert_vocab["，"] = "，"
    dict_daguan_vocab2bert_vocab["。"] = "。"
    dict_daguan_vocab2bert_vocab["！"] = "！"
    dict_daguan_vocab2bert_vocab["？"] = "？"

    print(dict_daguan_vocab2bert_vocab)

    for df_ in [df_train, df_val, df_test]:
        for text in df_['text']:
            text = text.split(" ")
            text_new = [dict_daguan_vocab2bert_vocab[w] for w in text if w in dict_daguan_vocab2bert_vocab]
            text_new = " ".join(text_new)
            assert "\t" not in text_new
            # print(text_new)
            # for char in text.split(" "):
            #     vocab_in_tasks[char] += 1

    for i in range(len(df_train)):
        text = df_train['text'][i]
        text = text.split(" ")
        text_new = [dict_daguan_vocab2bert_vocab[w] for w in text if w in dict_daguan_vocab2bert_vocab]
        text_new = " ".join(text_new)
        df_train.loc[i, "text"] = text_new

    output_dir = "/data2/code/DaguanFengxian/bert_model/data/splits_quchong/fold_" + split_num + "_" + model_type + "_vocab/"
    os.makedirs(output_dir, exist_ok=True)

    df_train.to_csv(
        os.path.join(output_dir, "train.txt"),
        index=False,
        sep="\t",
        header=None,
        encoding="utf-8"
    )

    for i in range(len(df_val)):
        text = df_val['text'][i]
        text = text.split(" ")
        text_new = [dict_daguan_vocab2bert_vocab[w] for w in text if w in dict_daguan_vocab2bert_vocab]
        text_new = " ".join(text_new)
        df_val.loc[i, "text"] = text_new

    df_val.to_csv(
        os.path.join(output_dir, "dev.txt"),
        index=False,
        sep="\t",
        header=None,
        encoding="utf-8",
    )

    for i in range(len(df_test)):
        text = df_test['text'][i]
        text = text.split(" ")
        text_new = [dict_daguan_vocab2bert_vocab[w] for w in text if w in dict_daguan_vocab2bert_vocab]
        text_new = " ".join(text_new)
        df_test.loc[i, "text"] = text_new
    df_test.to_csv(
        os.path.join(output_dir, "test.txt"),
        index=False,
        sep="\t",
        header=None,
        encoding="utf-8",
    )


if __name__ == '__main__':
    split_nums = ["0", "1", "2", "3", "4"]
    # model_types = ["albert_xxlarge", "bert", "bert_large"]
    # model_types = ["nezha_base", "nezha_large"]
    model_types = ["bert", "nezha_base"]

    for model_type in model_types:
        for split_num in split_nums:
            get_vocab_mapping(split_num, model_type)

    print("finish!")
