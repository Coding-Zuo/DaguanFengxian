# -*- coding:utf-8 -*-
import os
import numpy as np


def length_process(data_dir):
    train_dir = data_dir
    with open(train_dir, 'r', encoding='utf-8') as f:
        tmp_x = []

        for i, line in enumerate(f):
            if i == 0:
                continue

            sent_ = line.strip().split(",")[1]
            sent_ = sent_.split(" ")

            tmp_x.append(len(sent_))

        import matplotlib.pyplot as plt

        n, bins, patches = plt.hist(x=tmp_x, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('sentence length')
        plt.ylabel('Frequency')
        plt.title('Histogram: sentence length')
        # plt.text(23, 45, r'$\mu=15, b=3$')
        maxfreq = n.max()
        # 设置y轴的上限
        plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

        plt.show()


if __name__ == '__main__':
    length_process('/data2/nlpData/daguanfengxian/datagrand_2021_train.csv')
    length_process('/data2/nlpData/daguanfengxian/datagrand_2021_test.csv')
