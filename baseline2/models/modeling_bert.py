import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, BertConfig

from models.classifier import Classifier

from models.layers import AggregatorLayer
from models.layers import BiLSTMEncoder
from training.focal_loss import FocalLoss


class ClsBERT(BertPreTrainedModel):
    def __init__(self, config,
                 args,
                 label_list_level_1,
                 label_list_level_2,
                 label2freq_level_1,
                 label2freq_level_2,
                 ):
        super(ClsBERT, self).__init__(config)
        self.args = args
        self.args.hidden_size = config.hidden_size
        self.args.hidden_dim = config.hidden_size

        self.bert = BertModel(config=config)  # Load pretrained bert

        # TODO: 两层任务需要联合训练
        self.num_labels_level_1 = len(label_list_level_1)
        self.num_labels_level_2 = len(label_list_level_2)

        # if use_lstm: 添加一层lstm；
        self.lstm = None
        if self.args.use_lstm:
            self.lstm = BiLSTMEncoder(
                args,
            )

        # aggregator 层: 默认使用 BertPooler，如果指定用其他的aggregator，则添加
        self.aggregator_names = self.args.aggregator_names.split(",")
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
        self.class_weights_level_1 = F.softmax(torch.FloatTensor(
            class_weights_level_1
        ).to(self.args.device))

        class_weights_level_2 = []
        for i, lab in enumerate(label_list_level_2):
            class_weights_level_2.append(label2freq_level_2[lab])
        class_weights_level_2 = [1 / w for w in class_weights_level_2]
        if self.args.use_weighted_sampler:
            class_weights_level_2 = [math.sqrt(w) for w in class_weights_level_2]
        else:
            class_weights_level_2 = [w for w in class_weights_level_2]
        print("class_weights_level_2: ", class_weights_level_2)
        self.class_weights_level_2 = F.softmax(torch.FloatTensor(
            class_weights_level_2
        ).to(self.args.device))

    def forward(self, input_ids,
                attention_mask,
                token_type_ids,
                label_ids_level_1=None,
                label_ids_level_2=None,
                ):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
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
            if self.args.use_focal_loss:
                loss_fct = FocalLoss(
                    self.num_labels_level_2,
                    alpha=self.class_weights_level_2,
                    gamma=self.args.focal_loss_gamma,
                    size_average=True
                )
            elif self.args.use_class_weights:
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights_level_2)
            else:
                loss_fct = nn.CrossEntropyLoss()

            loss_level_2 = loss_fct(
                logits_level_2.view(-1, self.num_labels_level_2),
                label_ids_level_2.view(-1)
            )
            outputs = (loss_level_2,) + outputs

        return outputs
