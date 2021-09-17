# -*- coding:utf-8 -*-
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import BertEncoder
from models.layers import *
from transformers import BertPreTrainedModel, BertModel, BertConfig
from models.layers import *
from training.focal_loss import FocalLoss
from training.dice_loss import DiceLoss
from models.file_utils import cached_path
from pytorch_metric_learning.distances import DotProductSimilarity
from pytorch_metric_learning.losses import NTXentLoss, SupConLoss


class ClsBERTWithPABEE(BertPreTrainedModel):

    def __init__(self, config, args, label_list_level_1, label_list_level_2, label2freq_level_1, label2freq_level_2):
        super(ClsBERTWithPABEE, self).__init__(config)
        self.args = args
        self.config = config
        self.args.hidden_size = config.hidden_size
        self.args.hidddn_dim = config.hidden_size

        self.bert = BertModel(config=config)

        self.num_labels_level_1 = len(label_list_level_1)
        self.num_labels_level_2 = len(label_list_level_2)

        self.lstm = None
        self.gru = None
        if self.args.use_lstm:
            self.lstm = BiLSTMEncoder(args)
        if self.args.use_gru:
            self.gru = BiGRUEncoder(args)

        self.aggregator_names = self.args.aggregator.split(",")
        self.aggregator_names = [w.strip() for w in self.aggregator_names]
        self.aggregators = nn.ModuleList()
        for aggre_name in self.aggregator_names:
            if aggre_name == "bert_pooler":
                continue
            else:
                aggregator_op = AggregatorLayer(self.args, aggregator_name=aggre_name)
                self.aggregators.append(aggregator_op)

        if self.args.use_ms_dropout:
            self.classifier_level_2 = MultiSampleClassifier(args, input_dim=self.args.hidden_size,
                                                            num_labels=self.num_labels_level_2)
        else:
            self.classifier_level_2 = Classifier(
                args,
                input_dim=self.args.hidden_size,
                num_labels=self.num_labels_level_2,
            )

            # class weights
        class_weights_level_1 = []
        for i, lab in enumerate(label_list_level_1):
            class_weights_level_1.append(label2freq_level_1[lab])
        class_weights_level_1 = [1 / w for w in class_weights_level_1]
        if self.args.use_weighted_sampler:
            class_weights_level_1 = [math.sqrt(w) for w in class_weights_level_1]
        else:
            class_weights_level_1 = [w for w in class_weights_level_1]
        print("class_weights_level_1: ", class_weights_level_1)
        self.class_weights_level_1 = F.softmax(torch.FloatTensor(class_weights_level_1).to(self.args.device))

        class_weights_level_2 = []
        for i, lab in enumerate(label_list_level_2):
            class_weights_level_2.append(label2freq_level_2[lab])
        class_weights_level_2 = [1 / w for w in class_weights_level_2]
        if self.args.use_weighted_sampler:
            class_weights_level_2 = [math.sqrt(w) for w in class_weights_level_2]
        else:
            class_weights_level_2 = [w for w in class_weights_level_2]
        print("class_weights_level_2: ", class_weights_level_2)
        self.class_weights_level_2 = F.softmax(torch.FloatTensor(class_weights_level_2).to(self.args.device))

    def forward(self, input_ids, attention_mask, token_type_ids, label_ids_level_1=None, label_ids_level_2=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            output_hidden_states=True)
        sequence_output = outputs[0]
        bert_pooled_output = outputs[1]

        # 每层的隐含状态
        all_hidden_states = outputs[2]
        assert len(all_hidden_states) == self.config.num_hidden_layers + 1

        # 每层的池化结果：
        all_pooled_outputs = []
        for i in range(self.config.num_hidden_layers):
            hid_state = all_hidden_states[i + 1]
            if self.args.use_lstm:
                hid_state = self.lstm(hid_state)  # torch.Size([64, 132, 768])
            if self.args.use_gru:
                hid_state = self.gru(hid_state)  # torch.Size([64, 132, 768])

            bert_pooled_output = self.bert.pooler(hid_state)

            list_pooled_outpts = []
            if "bert_pooler" in self.aggregator_names:
                list_pooled_outpts.append(bert_pooled_output)
            for aggre_op in self.aggregators:
                pooled_outputs_ = aggre_op(hid_state, mask=attention_mask)
                list_pooled_outpts.append(pooled_outputs_)
            pooled_outputs = sum(list_pooled_outpts)
            all_pooled_outputs.append(pooled_outputs)

        # 每层的logits
        all_logits = []
        for i in range(self.config.num_hidden_layers):
            logits = self.classifier_level_2(all_pooled_outputs[i])
            all_logits.append(logits)
        outputs = (all_logits,)
        outputs = (all_logits[-1],) + outputs

        if label_ids_level_2 is not None:
            if self.args.use_class_weights:
                weight = self.class_weights_level_2
            else:
                weight = None
            if self.args.loss_fct_name == 'focal':
                loss_fct = FocalLoss(gamma=self.args.focal_loss_gamma, alpha=weight, reduction='mean')
            elif self.args.loss_fct_name == 'dice':
                loss_fct = DiceLoss(
                    with_logits=True,
                    smooth=1.0,
                    ohem_ratio=0.8,
                    alpha=0.01,
                    square_denominator=True,
                    index_label_position=True,
                    reduction="mean"
                )
            else:
                loss_fct = nn.CrossEntropyLoss(weight=weight)

            total_loss = None
            total_weights = 0

            for ix, logits_item in enumerate(all_logits):
                loss_ix = loss_fct(logits_item.view(-1, self.num_labels_level_2), label_ids_level_2.view(-1))
                if total_loss is None:
                    total_loss = loss_ix
                else:
                    total_loss += loss_ix * (ix + 1)
                total_weights += ix + 1
            outputs = (total_loss / total_weights,) + outputs
        return outputs
