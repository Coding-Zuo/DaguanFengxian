# -*- coding:utf-8 -*-
import json
import os


def vocab_process(data_dir, to_folder):
    vocab_file_level_1 = os.path.join(to_folder, 'labels_level_1.txt')
    vocab_file_level_2 = os.path.join(to_folder, 'labels_level_2.txt')

    label2freq_level_1_file = os.path.join(to_folder, 'label2freq_level_1.json')
    label2freq_level_2_file = os.path.join(to_folder, 'label2freq_level_2.json')

    with open(data_dir, 'r', encoding='utf-8') as f:
        vocab_level_1 = {}
        vocab_level_2 = {}

        for i, line in enumerate(f):
            if i == 0:
                continue

            label_ = line.strip().split(",")[-1]
            label_level_1 = label_.strip().split("-")[0]
            label_level_2 = label_

            if label_level_1 not in vocab_level_1:
                vocab_level_1[label_level_1] = 0
            vocab_level_1[label_level_1] += 1

            if label_level_2 not in vocab_level_2:
                vocab_level_2[label_level_2] = 0
            vocab_level_2[label_level_2] += 1

        json.dump(vocab_level_1, open(label2freq_level_1_file, 'w', encoding='utf-8'))
        json.dump(vocab_level_2, open(label2freq_level_2_file, 'w', encoding='utf-8'))

        vocab_level_1 = list(vocab_level_1.items())
        vocab_level_1 = sorted(vocab_level_1, key=lambda x: x[1], reverse=True)
        print("vocab_level_1: ", vocab_level_1)
        vocab_level_1 = [w[0] for w in vocab_level_1]

        vocab_level_2 = list(vocab_level_2.items())
        vocab_level_2 = sorted(vocab_level_2, key=lambda x: x[1], reverse=True)
        print("vocab_level_2: ", vocab_level_2)
        vocab_level_2 = [w[0] for w in vocab_level_2]

        with open(vocab_file_level_1, 'w', encoding='utf-8') as f_out:
            for lab in vocab_level_1:
                f_out.write(lab + '\n')

        with open(vocab_file_level_2, 'w', encoding='utf-8') as f_out:
            for lab in vocab_level_2:
                f_out.write(lab + '\n')


if __name__ == '__main__':
    data_dir = '/data2/code/DaguanFengxian/bert_model/data/datagrand_2021_train.csv'
    to_folder = '/data2/code/DaguanFengxian/bert_model/data/'
    vocab_process(data_dir, to_folder)
