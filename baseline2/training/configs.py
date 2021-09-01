# -*- coding: utf-8 -*-
from transformers import BertConfig, BertTokenizer

from transformers import BertConfig, BertTokenizer

from models.modeling_bert import ClsBERT

"""
python training/main.py --model_type bert --model_name_or_path /home/zuoyuhui/DataGame/haihuai_RC/chinese-bert-wwm-ext  --data_dir /data2/code/DaguanFengxian/bert_model/data/splits/fold_0_bertvocab  --label_file_level_1 /data2/code/DaguanFengxian/bert_model/data/labels_level_1.txt --label_file_level_2 /data2/code/DaguanFengxian/bert_model/data/labels_level_2.txt --task daguan --aggregator bert_pooler --model_dir /data2/code/DaguanFengxian/bert_model/data/outputs/bert_base_bertvocab  --do_train --do_eval --train_batch_size 16 --num_train_epochs 50 --embeddings_learning_rate 0.4e-4 --encoder_learning_rate 0.5e-4 --classifier_learning_rate 5e-4 --warmup_steps 200 --max_seq_len 132 --dropout_rate 0.15 --metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 200 --patience 12 --label2freq_level_1_dir /data2/code/DaguanFengxian/bert_model/data/label2freq_level_1.json --label2freq_level_2_dir /data2/code/DaguanFengxian/bert_model/data/label2freq_level_2.json
"""

MODEL_CLASSES = {
    'bert': (BertConfig, ClsBERT, BertTokenizer),
}

MODEL_PATH_MAP = {
    # 'bert': './bert_finetune_cls/resources/bert_base_uncased',
    'bert': './resources/bert/chinese-bert-wwm-ext_embedding_replaced_random',
    'albert': './resources/albert/albert_base_zh/albert_base_zh_embedding_replaced_random',
}
