# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AlbertModel, AlbertPreTrainedModel
from models.layers import *
from training.focal_loss import FocalLoss
from training.dice_loss import DiceLoss
from models.file_utils import cached_path
from pytorch_metric_learning.distances import DotProductSimilarity
from pytorch_metric_learning.losses import NTXentLoss, SupConLoss


class ClsALBERT(AlbertPreTrainedModel):
    def __init__(self, config,
                 args,
                 label_list_level_1,
                 label_list_level_2,
                 label2freq_level_1,
                 label2freq_level_2,
                 ):
        super(ClsALBERT, self).__init__(config)
        self.args = args
        self.args.hidden_size = config.hidden_size
        self.args.hidden_dim = config.hidden_size

        self.albert = AlbertModel(config=config)

        self.num_labels_level_1 = len(label_list_level_1)
        self.num_labels_level_2 = len(label_list_level_2)

        self.lstm = None
        if self.args.use_lstm:
            self.lstm = BiLSTMEncoder(args)

        # aggregator 层: 默认使用 BertPooler，如果指定用其他的aggregator，则添加
        self.aggregator_names = self.args.aggregator.split(",")
        self.aggregator_names = [w.strip() for w in self.aggregator_names]
        self.aggregator_names = [w for w in self.aggregator_names if w]
        self.aggregators = nn.ModuleList()
        for aggre_name in self.aggregator_names:
            if aggre_name == "bert_pooler":
                continue
            else:
                aggregator_op = AggregatorLayer(self.args, aggregator_name=aggre_name)
                self.aggregators.append(aggregator_op)

        # 分类层
        # self.classifier_level_1 = Classifier(
        #     args,
        #     input_dim=self.args.hidden_size * len(self.aggregator_names),
        #     num_labels=self.num_labels_level_1,
        # )

        if self.args.use_ms_dropout:
            self.classifier_level_2 = MultiSampleClassifier(
                args,
                input_dim=self.args.hidden_size,
                num_labels=self.num_labels_level_2,
            )
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

    def forward(self, input_ids,
                attention_mask,
                token_type_ids,
                label_ids_level_1=None,
                label_ids_level_2=None,
                ):
        outputs = self.albert(input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        bert_pooled_output = outputs[1]  # [CLS]

        list_pooled_outpts = []
        if "bert_pooler" in self.aggregator_names:
            list_pooled_outpts.append(bert_pooled_output)

        for aggre_op in self.aggregators:
            pooled_outputs_ = aggre_op(sequence_output, mask=attention_mask)
            list_pooled_outpts.append(pooled_outputs_)

        pooled_outputs = sum(list_pooled_outpts)

        # print("pooled_outputs: ", pooled_outputs.shape)

        # 分类层： level 2
        logits_level_2 = self.classifier_level_2(pooled_outputs)  # [bsz, self.num_labels_level_2]

        outputs = (logits_level_2,)  # add hidden states and attention if they are here

        # 1. loss
        if label_ids_level_2 is not None:
            if self.args.use_class_weights:
                weight = self.class_weights_level_2
            else:
                weight = None

            if self.args.loss_fct_name == "focal":
                loss_fct = FocalLoss(
                    gamma=self.args.focal_loss_gamma,
                    alpha=weight,
                    reduction="mean"
                )
            elif self.args.loss_fct_name == 'dice':
                loss_fct = DiceLoss(
                    with_logits=True,
                    smooth=1.0,
                    ohem_ratio=0,
                    alpha=0.01,
                    square_denominator=True,
                    index_label_position=True,
                    reduction="mean"
                )
            else:
                loss_fct = nn.CrossEntropyLoss(weight=weight)

            loss_level_2 = loss_fct(
                logits_level_2.view(-1, self.num_labels_level_2),
                label_ids_level_2.view(-1)
            )

            # 基于对比学习的损失计算
            if self.args.contrastive_loss != "None":
                if self.args.contrastive_loss == "ntxent_loss":  # InfoNCE
                    loss_fct_contrast = NTXentLoss(temperature=self.args.contrastive_temperature,
                                                   distance=DotProductSimilarity())
                elif self.args.contrastive_loss == "supconloss":
                    loss_fct_contrast = SupConLoss(temperature=self.args.contrastive_temperature,
                                                   distance=DotProductSimilarity())
                else:
                    raise ValueError("unsupported contrastive loss function: {}".format(self.args.contrastive_loss))

                if self.args.what_to_contrast == "sample":  # batch中的数据
                    embeddings = pooled_outputs
                    labels = label_ids_level_2.view(-1)
                elif self.args.what_to_contrast == "sample_and_class_embeddings":  # 把分类层的权重也加进来
                    embeddings = torch.cat([pooled_outputs, self.classifier_level_2.linear.weight], dim=0)
                    labels = torch.cat([label_ids_level_2.view(-1),
                                        torch.arange(0, self.num_labels_level_2).to(self.args.device)], dim=-1)
                else:
                    raise ValueError("unsupported contrastive features: {}".format(self.args.what_to_contrast))

                contra_loss_level_2 = loss_fct_contrast(embeddings, labels)
                # logger.info("ce loss: {}; contrastive loss: {}".format(
                #     loss_level_2, contra_loss_level_2
                # ))
                loss_level_2 = loss_level_2 + self.args.contrastive_loss_weight * contra_loss_level_2

            outputs = (loss_level_2,) + outputs

        return outputs
