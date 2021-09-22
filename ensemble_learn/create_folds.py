# -*- coding:utf-8 -*-
import pandas as pd
from sklearn import model_selection

if __name__ == '__main__':
    df = pd.read_csv("/data2/code/DaguanFengxian/ensemble_learn/word2vec-nlp-tutorial/labeledTrainData.tsv", sep="\t")
    print(df.head())
    df.loc[:, "kfold"] = -1
    print(df.head())
    df = df.sample(frac=1).reset_index(drop=True)

    y = df.sentiment.values
    skf = model_selection.StratifiedKFold(n_splits=5)

    for f, (t_, v_) in enumerate(skf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = int(f)

    print(df.head())
    print(df.tail())
    df.to_csv("/data2/code/DaguanFengxian/ensemble_learn/train_folds.csv", index=False)
