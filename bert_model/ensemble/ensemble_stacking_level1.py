# -*- coding:utf-8 -*-
"""
    IMPORTING LIBS
"""
import numpy as np
import pandas as pd
import os
import logging

import torch
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

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_root_path = "/data2/code/DaguanFengxian/bert_model/data/outputs/"

bert120k_5folds = [
    "new_ensemble/train.bert120k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold0__v3",
    "new_ensemble/train.bert120k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold1__v3",
    "new_ensemble/train.bert120k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold2__v3",
    "new_ensemble/train.bert120k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold3__v3",
    "new_ensemble/train.bert120k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold4__v3",
]

bert_wwm_5folds = [
    "ensemble/train.bert_base_wwm_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold0__v3",
    "ensemble/train.bert_base_wwm_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold1__v3",
    "ensemble/train.bert_base_wwm_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold2__v3",
    "ensemble/train.bert_base_wwm_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold3__v3",
    "ensemble/train.bert_base_wwm_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold4__v3",
]

nezha_5folds = [
    "ensemble/train.nezha_base_wwm_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold0_v3",
    "ensemble/train.nezha_base_wwm_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold1__v3",
    "ensemble/train.nezha_base_wwm_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold2__v3",
    "ensemble/train.nezha_base_wwm_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold3__v3",
    "ensemble/train.nezha_base_wwm_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold4__v3",
]

albert_5folds = [
    "new_ensemble/train.nezha_base_wwm_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold0_v3",
    "new_ensemble/train.nezha_base_wwm_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold1__v3",
    "new_ensemble/train.nezha_base_wwm_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold2__v3",
    "new_ensemble/train.nezha_base_wwm_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold3__v3",
    "new_ensemble/train.nezha_base_wwm_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold4__v3",
]

new_nezha_4folds = [
    "4zhe/train.newnezha_dice_supcon_lr1e-5_nomulti_fgm__fold0__v3",
    "4zhe/train.newnezha_dice_supcon_lr1e-5_nomulti_fgm__fold1__v3",
    "4zhe/train.newnezha_dice_supcon_lr1e-5_nomulti_fgm__fold2__v3",
    "4zhe/train.newnezha_dice_supcon_lr1e-5_nomulti_fgm__fold3__v3",
]

bert300_4folds = [
    "4zhe/train.newbert300_dice_supcon_lr1e-5_57__fold0__v3",
    "4zhe/train.newbert300_dice_supcon_lr1e-5_57__fold1__v3",
    "4zhe/train.newbert300_dice_supcon_lr1e-5_57__fold2__v3",
    "4zhe/train.newbert300_dice_supcon_lr1e-5_57__fold3__v3",
]

bert150k_4folds = [
    "4zhe/train.newbert300_dice_supcon_lr1e-5_57__fold0__v3",
    "4zhe/train.newbert300_dice_supcon_lr1e-5_57__fold1__v3",
    "4zhe/train.newbert300_dice_supcon_lr1e-5_57__fold2__v3",
    "4zhe/train.newbert300_dice_supcon_lr1e-5_57__fold3__v3",
]


def full_args(args):
    if not hasattr(args, 'use_ms_dropout'):
        args.use_ms_dropout = False
    if not hasattr(args, 'use_hongfan'):
        args.use_hongfan = False
    if not hasattr(args, 'contrastive_loss'):
        args.contrastive_loss = None
    if not hasattr(args, 'use_focal_loss'):
        args.use_focal_loss = False
    if not hasattr(args, 'use_freelb'):
        args.use_freelb = False
    if not hasattr(args, 'use_multi_task'):
        args.use_multi_task = False
    if not hasattr(args, 'use_gru'):
        args.use_gru = False
    if not hasattr(args, 'use_swa'):
        args.use_swa = False


from sklearn.linear_model import LogisticRegression

"""
    level 1
"""
all_model = [new_nezha_4folds, new_nezha_4folds1]
final_learner = LogisticRegression()
train_meta_prob = None  # [14009,35*k]
test_meta_prob = None  # [6004,35*k]

for idx, model_kfolds in enumerate(all_model):
    model_folds_pred_test_sum = None  # [6004,35]
    model_retrain_dict_dev = dict()  # [14009,35]
    tokenizer = None
    args = None
    for checkpoint in model_kfolds:
        logger.info("********** start:" + checkpoint + "**********")
        checkpoint_path = model_root_path + checkpoint
        args = torch.load(os.path.join(checkpoint_path + '/training_args_bin'))
        full_args(args)

        tokenizer = MODEL_CLASSES[args.model_encoder_type][2].from_pretrained(args.encoder_name_or_path)
        """
        train：14009 2801.8
        test:6004
        train_dataset:  11188
        train_sample_weights:  11188
        dev_dataset:  2821
        dev_sample_weights:  2821
        test_dataset:  6004
        test_sample_weights:  6004
        """
        dev_dataset, dev_sample_weights = load_and_cache_examples(args, tokenizer, mode="dev")
        test_dataset, test_sample_weights = load_and_cache_examples(args, tokenizer, mode="test")

        trainer = Trainer(
            args,
            dev_dataset=dev_dataset,
            test_dataset=test_dataset,
            dev_sample_weights=dev_sample_weights,
            test_sample_weights=test_sample_weights,
        )
        trainer.load_model()
        # 预测类别idx ，             [logits, idx]        prob_array [2821,35]    {id:prob_array(35) }  all_logits
        preds_label_idx_dev, logits_label_idx_dev, preds_prob_array_dev, id_prob_dict_dev, logits_array_dev = trainer.ensemble_test(
            "dev")
        preds_label_idx_test, logits_label_idx_test, preds_prob_array_test, id_prob_dict_test, logits_array_test = trainer.ensemble_test(
            "test")

        # 去重
        for ids, prob in id_prob_dict_dev.items():
            if ids in model_retrain_dict_dev: continue
            model_retrain_dict_dev[ids] = prob

        # test加和
        if model_folds_pred_test_sum is None:
            model_folds_pred_test_sum = np.array(preds_prob_array_test)
        else:
            model_folds_pred_test_sum += np.array(preds_prob_array_test)

    if train_meta_prob is None:
        train_meta_prob = model_retrain_dict_dev
    else:
        for id_, probs in model_retrain_dict_dev.items():
            a = np.hstack((train_meta_prob[id_], probs))
            train_meta_prob[id_] = a

    if test_meta_prob is None:  # 取平均
        test_meta_prob = model_folds_pred_test_sum / len(model_kfolds)
    else:
        b = model_folds_pred_test_sum / len(model_kfolds)
        test_meta_prob = np.hstack((test_meta_prob, b))

# 给每个特征加标签，去train里找
train_file = '/data2/code/DaguanFengxian/bert_model/data/datagrand_2021_train.csv'
label_dict = {}
true_labels = get_labels("/data2/code/DaguanFengxian/bert_model/data/labels_level_2.txt")
with open(train_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i == 0:
            continue
        id_ = line.strip().split(",")[0]
        label_ = line.strip().split(",")[-1]
        label_dict[int(id_)] = true_labels.index(label_)

# 处理train_meta_prob{id: [id_prob_array1,id_prob_array2]}   test_meta_prob [prob_array1,prob_array2]
train_meta_prob_new = None
train_meta_label = []
for id_, prob_list in train_meta_prob.items():
    train_meta_label.append(label_dict[id_])
    if train_meta_prob_new is None:
        train_meta_prob_new = prob_list
    else:
        train_meta_prob_new = np.vstack((train_meta_prob_new, prob_list))

# tocsv
output_dir = "/data2/code/DaguanFengxian/bert_model/data/ensemble_data/"
pd.DataFrame(train_meta_prob_new).to_csv(  # [14009, 35*k]
    os.path.join(output_dir, "train_meta_prob.csv"),
    index=False,
)
pd.DataFrame(train_meta_label).to_csv(  # [14009, 35*k]
    os.path.join(output_dir, "train_meta_label.csv"),
    index=False,
)
pd.DataFrame(test_meta_prob).to_csv(  # [6004,35*k]
    os.path.join(output_dir, "test_meta_prob.csv"),
    index=False,
)

"""
    level 2
"""
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression, LinearRegression

xgb_cls = xgb.XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=100,
                            objective="multi:softprob", num_class=35,
                            subsample=0.8, colsample_bytree=0.8, tree_method='gpu_hist',
                            min_child_samples=3, eval_metric='auc', reg_lambda=0.5)

# print(f"Train accuracy: {final_learner.score(test_meta_model, true_labels)}")
