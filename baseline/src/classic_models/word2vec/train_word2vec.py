# -*- coding:utf-8 -*-
import os
from gensim.models import Word2Vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class MySentence(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            print(fname)
            for line in open(os.path.join(self.dirname, fname), 'r', encoding='utf-8'):
                yield line.split()


def train_w2v_model(model_save_dir, sentences: MySentence):
    print("Start...")
    model = Word2Vec(
        sentences,
        sg=0,
        hs=1,
        size=1024,
        window=12,
        min_count=1,
        workers=8,
        iter=30
    )

    print(model.max_final_vocab)
    model.wv.save_word2vec_format(model_save_dir, binary=False)
    print('Finished!')


if __name__ == '__main__':

    sentences_dir = '/data2/nlpData/daguanfengxian/wujiandu/txt_format/'
    model_save_dir = '/data2/nlpData/daguanfengxian/word2vec/dim_1024_sg_0_hs_1_epochs_30/w2v.vectors'
    sentence = MySentence(sentences_dir)

    train_w2v_model(model_save_dir, sentence)
