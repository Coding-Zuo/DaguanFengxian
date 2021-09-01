# -*- coding:utf-8 -*-
import json
import threading
import tqdm

from multiprocessing import Process
# json line格式改为text格式

# 14939045 行

# 2.无标注文本：大规模无标注的文本规模是亿级，
#    可供选手选择用来进行语言模型训练，多行格式，
#    每行为json表示，包含title和content两个字段，
#    采用统一的脱敏编码形式；
from src.classic_models.uitls.text_utils import split_sent


def jsonline2txt(from_json_dir, to_txt_dir, thread_idx, num_threads):
    with open(from_json_dir, 'r', encoding='utf-8') as f_in, \
            open(to_txt_dir, 'w', encoding='utf-8', buffering=20480) as f_out:
        for i, line in tqdm.tqdm(enumerate(f_in)):
            if i % num_threads != thread_idx:
                continue
            line = line.strip()
            if not line:
                continue

            line = json.loads(line)

            title_ = line['title'].strip()
            content_ = line['content'].strip()

            if title_:
                f_out.write(title_ + "\n");
            if content_:
                list_sents = split_sent(content_, spliter="。？?！")
                if list_sents:
                    for sent in list_sents:
                        sent = sent.strip()
                        f_out.write(sent + "\n")
            f_out.write("\n")


if __name__ == '__main__':
    from_json_dir = "/data2/nlpData/daguanfengxian/datagrand_2021_unlabeled_data.json"

    num_threads = 16
    for i in range(num_threads):
        to_txt_dir_ = '/data2/nlpData/daguanfengxian/wujiandu/txt_format/unlabeled_corpus_%d.txt' % i

        # p_ = Process(
        #     target=jsonline2txt,
        #     args=(from_json_dir, to_txt_dir_, i, num_threads)
        # )

        p_ = threading.Thread(
            target=jsonline2txt,
            args=(from_json_dir, to_txt_dir_, i, num_threads),
        )
        p_.start()
