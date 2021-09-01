# -*- coding:utf-8 -*-
import argparse
import os
import re
import torch
import numpy as np
from transformers import BertConfig, BertTokenizer, BertModel, AlbertModel
import sys

sys.path.append("./")

from models.model_utils import get_embedding_matrix_and_vocab
from models.model_envs import MODEL_CLASSES


def replace_albert_embeddings_random(pretrained_model_path, new_model_path, w2v_file, embedding_dim=768,
                                     model_type="bert"):
    vocab_list, _ = get_embedding_matrix_and_vocab(w2v_file, include_special_tokens=False)

    # 加载预训练模型
    tokenizer = MODEL_CLASSES[model_type][2].from_pretrained(pretrained_model_path)
    config_class, model_class, _ = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(pretrained_model_path)
    model = model_class.from_pretrained(pretrained_model_path, config=config)

    bert_embed_matrix = model.embeddings.word_embeddings.weight.detach().cpu().numpy().tolist()
    bert_vocab = tokenizer.get_vocab()

    print(type(bert_vocab))
    print(len(bert_vocab))
    print(bert_vocab['[PAD]'])

    # 构建新的vocab
    new_vocab_list, new_vector_list = [], []
    # 将[PAD] [UNK] [CLS] [SEP] [MASK] 的embedding加入
    for w in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]:
        new_vocab_list.append(w)
        new_vector_list.append(bert_embed_matrix[bert_vocab[w]])

    for w_ in vocab_list:
        if not re.search("[0-9]", w_):
            print("non indexed word: ", w_)
            new_vocab_list.append(w_)
            new_vector_list.append(bert_embed_matrix[bert_vocab[w_]])
        else:
            new_vocab_list.append(w_)
            new_vector_list.append((np.random.randn(embedding_dim).astype(np.float32) * 0.2).tolist())

    assert len(new_vocab_list) == len(new_vector_list)

    vocab_file = os.path.join(new_model_path, "vocab.txt")
    with open(vocab_file, "w", encoding="utf-8") as f:
        for w in new_vocab_list:
            f.write(w + "\n")

    config.vocab_size = len(new_vocab_list)
    config.save_pretrained(new_model_path)

    model.embeddings.word_embeddings.weight = torch.nn.Parameter(torch.FloatTensor(new_vector_list))
    model.save_pretrained(new_model_path)


def replace_albert_embeddings(pretrained_model_path, new_model_path, w2v_file, embedding_dim=768,
                              model_type="bert"):
    vocab_list, vector_list = get_embedding_matrix_and_vocab(w2v_file, include_special_tokens=False)

    tokenizer = MODEL_CLASSES[model_type][2].from_pretrained(pretrained_model_path)
    config_class, model_class, _ = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(pretrained_model_path)
    model = model_class.from_pretrained(pretrained_model_path, config=config)

    bert_embed_matrix = model.embeddings.word_embeddings.weight.detach().cpu().numpy().tolist()
    bert_vocab = tokenizer.get_vocab()
    print(type(bert_vocab))
    print(len(bert_vocab))
    print(bert_vocab["[PAD]"])

    # 构建新的vocab
    new_vocab_list, new_vector_list = [], []
    # 将[PAD], [UNK], [CLS], [SEP], [MASK] 的embedding加入
    for w in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]:
        new_vocab_list.append(w)
        new_vector_list.append(bert_embed_matrix[bert_vocab[w]])

    for w_, vec_ in zip(vocab_list, vector_list):
        if not re.search("[0-9]", w_):
            print("non indexed word: ", w_)
            new_vocab_list.append(w_)
            new_vector_list.append(bert_embed_matrix[bert_vocab[w_]])
        else:
            new_vocab_list.append(w_)
            new_vector_list.append(vec_)

    assert len(new_vocab_list) == len(new_vector_list)

    vocab_file = os.path.join(new_model_path, "vocab.txt")
    with open(vocab_file, "w", encoding="utf-8") as f:
        for w in new_vocab_list:
            f.write(w + "\n")

    config.vocab_size = len(new_vocab_list)
    config.save_pretrained(new_model_path)

    model.embeddings.word_embeddings.weight = torch.nn.Parameter(torch.FloatTensor(new_vector_list))
    model.save_pretrained(new_model_path)


if __name__ == '__main__':
    # for bert base, 随机初始化矩阵替换embedding
    pretrained_model_path = "/home/zuoyuhui/DataGame/haihuai_RC/chinese-bert-wwm-ext"
    w2v_file = "/data2/nlpData/daguanfengxian/word2vec/dim_768_sg_0_hs_1_epochs_30/w2v.vectors"
    new_model_path = "/data2/code/DaguanFengxian/bert_model/data/mybert/bert-base-chinese_embedding_replaced_random"

    os.makedirs(new_model_path, exist_ok=True)
    replace_albert_embeddings_random(pretrained_model_path,
                                     new_model_path,
                                     w2v_file,
                                     embedding_dim=768,
                                     model_type="bert")

    # bert base： 训练好的w2v替换bert原本的embedding
    pretrained_model_path = "/home/zuoyuhui/DataGame/haihuai_RC/chinese-bert-wwm-ext"
    w2v_file = "/data2/nlpData/daguanfengxian/word2vec/dim_768_sg_0_hs_1_epochs_30/w2v.vectors"
    new_model_path = "/data2/code/DaguanFengxian/bert_model/data/mybert/bert-base-chinese_embedding_replaced_w2v"

    os.makedirs(new_model_path, exist_ok=True)
    replace_albert_embeddings(pretrained_model_path,
                              new_model_path,
                              w2v_file,
                              model_type="bert")
