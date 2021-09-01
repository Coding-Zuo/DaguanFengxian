# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from src.classic_models.models.layers import *
from src.classic_models.training.focal_loss import *


class ClsModel(nn.Module):
    def __init__(self, args, label_list_level_1, label_list_level_2):
        super(ClsModel, self).__init__()
        self.args = args

        # TODO: 两层任务要联合训练
        self.num_labels_level_1 = len(label_list_level_1)
        self.num_labels_level_2 = len(label_list_level_2)

        # embeddding层
        self.embeddings = EmbeddingLayer(args)

        # encoder层
        if args.encoder == 'textcnn':
            self.encoder = TextCnnEncoder(args)
        elif args.encoder == 'lstm':
            self.encoder = BiLSTMEncoder(args)
        else:
            raise ValueError("un-supported encoder type: {}".format(args.encoder))

        # aggregator层
        self.aggregator = AggregatorLayer(args)

        # 分类层
        self.classifier = Classifier(args, input_dim=args.hidden_dim, num_labels=self.num_labels_level_2)

        # class weight
        self.class_weights_level_2 = None
        if self.args.class_weights_level_2:
            self.class_weights_level_2 = self.args.class_weights_level_2.split(",")
        else:
            self.class_weights_level_2 = [1] * self.num_labels_level_2
        self.class_weights_level_2 = [float(w) for w in self.class_weights_level_2]
        self.class_weights_level_2 = torch.FloatTensor(self.class_weights_level_2).to(args.device)

    def forward(self, input_ids, attention_mask, position_ids=None, label_ids_level_1=None, label_ids_level_2=None,
                **kwargs):
        input_tensors = self.embeddings(input_ids)

        output_tensors = self.encoder(input_tensors)

        pooled_outputs = self.aggregator(output_tensors, mask=attention_mask)

        logits_level_2 = self.classifier(pooled_outputs)

        outputs = (logits_level_2,)

        if label_ids_level_2 is not None:
            if self.args.use_focal_loss:
                loss_fct = FocalLoss(self.num_labels_level_2, alpha=self.class_weights_level_2,
                                     gamma=self.args.focal_loss_gamma, size_average=True)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss_level_2 = loss_fct(logits_level_2.view(-1, self.num_labels_level_2), label_ids_level_2.view(-1))
            outputs = (loss_level_2,) + outputs
        return outputs
