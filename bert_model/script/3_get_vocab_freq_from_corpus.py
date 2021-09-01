# -*- coding:utf-8 -*-
import json
import os
import sys
from collections import defaultdict
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

sys.path.insert(0, './')  # 新添加的目录会优先于其他目录被import检查，程序退出后失效
"""
从开源语料库中获取token对应的频率
"""


class SentenceIter(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for i, fname in enumerate(os.listdir(self.dirname)):
            print(fname)
            for line in open(os.path.join(self.dirname, fname), 'r', encoding='utf-8'):
                yield line.strip()


def get_vocab_freq(sentence_iter, tokenizer):
    dict_vocab2freq = defaultdict(list)
    for i, sent in tqdm(enumerate(sentence_iter)):
        if not sent: continue

        tokens = tokenizer.tokenize(sent)
        for tok in tokens:
            dict_vocab2freq[tok] += 1

    return dict_vocab2freq


if __name__ == '__main__':
    chinese_bert_wwm_ext = "/home/zuoyuhui/DataGame/haihuai_RC/chinese-bert-wwm-ext"
    tokenizer = BertTokenizer.from_pretrained(chinese_bert_wwm_ext)

    corpos_folder = "/data2/code/DaguanFengxian/bert_model/data/mybert/news_corpus"
    sentence_iter = SentenceIter(corpos_folder)

    dict_vocab2freq = get_vocab_freq(sentence_iter, tokenizer)

    vocab2freq_path = "/data2/code/DaguanFengxian/bert_model/vocab_process/dict_vocab2freq_wwm_bertbase.json"
    json.dump(
        dict_vocab2freq,
        open(vocab2freq_path, "w", encoding='utf-8'),
        ensure_ascii=False
    )
