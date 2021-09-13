# -*- coding:utf-8 -*-
import pandas as pd
import glob
import numpy as np
from sklearn import metrics
from scipy.optimize import fmin
from functools import partial

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler


def run_training(pred_df, fold):
    train_df = pred_df[pred_df.kfold != fold].reset_index(drop=True)
    valid_df = pred_df[pred_df.kfold == fold].reset_index(drop=True)

    xtrian = train_df[['lr_pred', 'lr_cnt_pred', 'rf_svd_pred']].values
    xvalid = valid_df[['lr_pred', 'lr_cnt_pred', 'rf_svd_pred']].values

    scl = StandardScaler()
    xtrain = scl.fit_transform(xtrian)
    xvalid = scl.transform(xvalid)

    opt = LogisticRegression()
    opt.fit(xtrian, train_df.sentiment.values)
    preds = opt.predict_proba(xvalid)[:, 1]
    auc = metrics.roc_auc_score(valid_df.sentiment.values, preds)
    print(f"{fold},{auc}")
    valid_df.loc[:, "opt_pred"] = preds
    return opt.coef_


if __name__ == '__main__':
    files = glob.glob("/data2/code/DaguanFengxian/ensemble_learn/model_pred/*.csv")
    df = None
    for f in files:
        if df is None:
            df = pd.read_csv(f)
        else:
            temp_df = pd.read_csv(f)
            df = df.merge(temp_df, on='id', how='left')

    print(df.head(10))
    target = df.sentiment.values
    pred_cols = ['lr_pred', 'lr_cnt_pred', 'rf_svd_pred']

    coefs = []
    for j in range(5):
        coefs.append(run_training(df, j))

    coefs = pd.array(coefs)
    # print(metrics.roc_auc_score(preds_df.sentiment.values, preds_df.opt_pred.values))
    print(coefs)
    coefs = np.mean(coefs, axis=0)
    print(coefs)

    # wt_avg = coefs[0] * df.lr_pred.values + coefs[1] * df.lr_cnt_pred.values + coefs[2] * df.rf_svd_pred.values
    wt_avg = coefs[0][0] * df.lr_pred.values + coefs[0][1] * df.lr_cnt_pred.values + coefs[0][2] * df.rf_svd_pred.values
    print("optimal acu after finding coefs")
    print(metrics.roc_auc_score(target, wt_avg))
