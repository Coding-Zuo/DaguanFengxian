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

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
model_root_path = "/data2/code/DaguanFengxian/bert_model/data/outputs/"

"""
    level 2
"""
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

data_dir = "/data2/code/DaguanFengxian/bert_model/data/ensemble_data/"

train_meta_prob = pd.read_csv(data_dir + "train_meta_prob.csv")
train_meta_label = pd.read_csv(data_dir + "train_meta_label.csv")
test_meta_prob = pd.read_csv(data_dir + "test_meta_prob.csv")

train_meta_prob = np.array(train_meta_prob).astype(float)
train_meta_label = np.array(train_meta_label).astype(float).squeeze()
test_meta_prob = np.array(test_meta_prob).astype(float)

# label_list = get_labels("/data2/code/DaguanFengxian/bert_model/data/labels_level_2.txt")
# num_labels = len(label_list)
# xgb_cls = xgb.XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=100,
#                             objective="multi:softprob", num_class=35,
#                             subsample=0.8, colsample_bytree=0.8, tree_method='gpu_hist',
#                             min_child_samples=3, eval_metric='auc', reg_lambda=0.5)
# xgb_cls.fit(train_meta_prob, train_meta_label)
# xgb_cls.predict(test_meta_prob)
X_train, X_test, y_train, y_test = train_test_split(train_meta_prob, train_meta_label, test_size=0.2, random_state=41)

label_list = get_labels("/data2/code/DaguanFengxian/bert_model/data/labels_level_2.txt")
num_labels = len(label_list)


def objective(trial):
    x_train1, x_test1, y_train1, y_test1 = X_train, X_test, y_train, y_test
    params = {
        'tree_method': 'gpu_hist',
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate',
                                                   [0.008, 0.009, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]),
        'n_estimators': trial.suggest_categorical('n_estimators',
                                                  [10, 20, 50, 100, 200, 500]),
        'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17, 20]),
        'random_state': trial.suggest_categorical('random_state', [24, 48, 2020]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300)
    }

    xgb_cls = xgb.XGBClassifier(**params)
    xgb_cls.fit(X_train, y_train, eval_set=[(x_test1, y_test1)], early_stopping_rounds=50, verbose=False)
    preds = xgb_cls.predict(X_test)
    accuracy = balanced_accuracy_score(y_test, preds)
    return accuracy


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print("Number of finished trials: {}".format(len(study.trials)))
print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

params = {
    'lambda': 0.04564015476537224,
    'alpha': 2.2405137008670026,
    'colsample_bytree': 0.9,
    'subsample': 0.4,
    'learning_rate': 0.012,
    'n_estimators': 200,
    'max_depth': 13,
    'random_state': 48,
    'min_child_weight': 45,
    'num_class': 35,
    # 'eval_metric' :'auc',
}
# xgb_clf = xgb.XGBClassifier(**params, tree_method='gpu_hist')
xgb_clf = xgb.XGBClassifier(**trial.params, num_labels=35, tree_method='gpu_hist')
xgb_clf.fit(train_meta_prob, train_meta_label)
preds = xgb_clf.predict(test_meta_prob)
list_preds = map(int, preds.tolist())
print(list_preds)

from sklearn import linear_model

clf = linear_model.LogisticRegression(penalty='l2', C=35, multi_class='ovr')
clf.fit(train_meta_prob, train_meta_label)
pred_lg = clf.predict_proba(test_meta_prob)
pred_lg = np.argmax(pred_lg, axis=1)

"""
    save
"""

f_out = open(os.path.join("/data2/code/DaguanFengxian/bert_model/data/ensemble_data/stacking_xgb_4.csv"),
             "w", encoding="utf-8")
f_out.write("id,label" + "\n")
for i, pred_label_id in enumerate(list_preds):
    pred_label_name_level_2 = label_list[pred_label_id]
    f_out.write("%s,%s" % (str(i), str(pred_label_name_level_2)) + "\n")

f_out = open(os.path.join("/data2/code/DaguanFengxian/bert_model/data/ensemble_data/stacking_lg_4.csv"),
             "w", encoding="utf-8")
f_out.write("id,label" + "\n")
for i, pred_label_id in enumerate(pred_lg):
    pred_label_name_level_2 = label_list[pred_label_id]
    f_out.write("%s,%s" % (str(i), str(pred_label_name_level_2)) + "\n")
