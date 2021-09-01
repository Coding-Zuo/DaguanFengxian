# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import json, math
from models.layers import *


class DaguanModel(nn.Module):

    def __init__(self, config):
        super(DaguanModel, self).__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim

        self.num_labels_level_1 = config.num_labels_level_1
        self.num_labels_level_2 = config.num_labels_level_2

        self.lstm = None
        if config.use_lstm:
            self.lstm = BiLSTMEncoder(config)

        # aggregator 层: 默认使用 BertPooler，如果指定用其他的aggregator，则添加
        self.aggregator_names = config.aggregator.split(",")
        self.aggregator_names = [w.strip() for w in self.aggregator_names]
        self.aggregator_names = [w for w in self.aggregator_names if w]
        self.aggregators = nn.ModuleList()
        for aggre_name in self.aggregator_names:
            if aggre_name == "bert_pooler":
                continue
            else:
                aggregator_op = AggregatorLayer(config, aggregator_name=aggre_name)
                self.aggregators.append(aggregator_op)

        self.classifier_level_2 = Classifier(config, input_dim=self.hidden_dim, num_labels=self.num_labels_level_2)

    def forward(self, batch):
        attention_mask = batch[1]  # [bs, seq_len]
        context_encoding = batch[-2]  # [bs,seq_len, embed_dim]
        bert_pooled = batch[-1]  # [bs, embed_dim]

        list_pooled_ouputs = []
        if "bert_pooler" in self.aggregator_names:
            list_pooled_ouputs.append(bert_pooled)

        for aggre_op in self.aggregators:
            pooled_outputs_ = aggre_op(context_encoding, mask=attention_mask)
            list_pooled_ouputs.append(pooled_outputs_)

        pooled_outputs = sum(list_pooled_ouputs)  # [bs, embed_dim]

        logits_level_2 = self.classifier_level_2(pooled_outputs)  # [bs, 35]

        outputs = (logits_level_2,)
        return outputs
