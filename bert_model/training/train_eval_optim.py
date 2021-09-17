# -*- coding:utf-8 -*-
import torch
import json, math
import numpy as np
import os
import shutil
import collections
import logging
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from transformers import AdamW
from sklearn.metrics import classification_report
from tensorboardX import SummaryWriter
from dataload.data_loader_bert import get_labels
from training.focal_loss import FocalLoss

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

"""
    train_eval_pipline
"""

"""
    compute_metrics
"""


def compute_metrics(intent_preds, intent_labels):
    assert len(intent_preds) == len(intent_labels)
    results = {}
    classification_report_dict = classification_report(intent_preds, intent_labels, output_dict=True)
    for key0, val0 in classification_report_dict.items():
        if isinstance(val0, dict):
            for key1, val1 in val0.items():
                results[key0 + "__" + key1] = val1

        else:
            results[key0] = val0
    return results


"""
    eval
"""


@torch.no_grad()
def eval_model(args, encoder, model, dataloader, mode, label_list_level_2, class_weights_level_1,
               class_weights_level_2):
    # Eval!
    logger.info("***** Running evaluation on %s dataset *****", mode)
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds_level_1 = None
    preds_level_2 = None
    out_label_ids_level_1 = None
    out_label_ids_level_2 = None

    encoder.eval()
    model.eval()

    for batch in tqdm(dataloader, desc="Evaluating"):
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2], }  # XLM don't use segment_ids
        label_ids_level_2 = batch[4]

        outputs = encoder(**inputs)
        batch.append(outputs[0])
        batch.append(outputs[1])

        logits_level_2 = model(batch)[0]

        loss = compute_loss(args, class_weights_level_1, class_weights_level_2, logits_level_2, label_ids_level_2)
        eval_loss += loss.mean().item()
        del batch

        nb_eval_steps += 1

        # label prediction
        if preds_level_2 is None:
            preds_level_2 = logits_level_2.detach().cpu().numpy()
            out_label_ids_level_2 = label_ids_level_2.detach().cpu().numpy()
        else:
            preds_level_2 = np.append(preds_level_2, logits_level_2.detach().cpu().numpy(), axis=0)
            out_label_ids_level_2 = np.append(out_label_ids_level_2, label_ids_level_2.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    results = {
        'loss': eval_loss
    }

    # label prediction result
    preds_level_2 = np.argmax(preds_level_2, axis=1)

    results_level_2 = compute_metrics(preds_level_2, out_label_ids_level_2)
    for key_, val_ in results_level_2.items():
        results[key_ + "__level_2"] = val_

    logger.info("***** Eval results *****")
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    # 将预测结果写入文件
    if mode == "test":
        f_out = open(os.path.join(args.exp_name, "test_predictions.csv"), "w", encoding="utf-8")
        f_out.write("id,label" + "\n")

        list_preds_level_2 = preds_level_2.tolist()
        for i, pred_label_id in enumerate(list_preds_level_2):
            pred_label_name_level_2 = label_list_level_2[pred_label_id]
            f_out.write("%s,%s" % (str(i), str(pred_label_name_level_2)) + "\n")

    return results


"""
    loss
"""


def get_class_weigth(args):
    label_list_level_1 = get_labels(args.label_file_level_1)
    label_list_level_2 = get_labels(args.label_file_level_2)

    label2freq_level_1 = json.load(open(args.label2freq_level_1, "r", encoding="utf-8"))
    label2freq_level_2 = json.load(open(args.label2freq_level_2, "r", encoding="utf-8"))

    # class weights
    class_weights_level_1 = []
    for i, lab in enumerate(label_list_level_1):
        class_weights_level_1.append(label2freq_level_1[lab])
    class_weights_level_1 = [1 / w for w in class_weights_level_1]

    if args.use_weighted_sampler:
        class_weights_level_1 = [math.sqrt(w) for w in class_weights_level_1]
    else:
        class_weights_level_1 = [w for w in class_weights_level_1]
    print("class_weights_level_1: ", class_weights_level_1)
    class_weights_level_1 = F.softmax(torch.FloatTensor(class_weights_level_1)).to(args.device)

    class_weights_level_2 = []
    for i, lab in enumerate(label_list_level_2):
        class_weights_level_2.append(label2freq_level_2[lab])
    class_weights_level_2 = [1 / w for w in class_weights_level_2]
    if args.use_weighted_sampler:
        class_weights_level_2 = [math.sqrt(w) for w in class_weights_level_2]
    else:
        class_weights_level_2 = [w for w in class_weights_level_2]
    print("class_weights_level_2: ", class_weights_level_2)
    class_weights_level_2 = F.softmax(torch.FloatTensor(class_weights_level_2)).to(args.device)
    return label_list_level_1, label_list_level_2, class_weights_level_1, class_weights_level_2


def compute_loss(args, class_weights_level_1, class_weights_level_2, logits_level_2, label_ids_level_2):
    if args.use_focal_loss:
        loss_fct = FocalLoss(
            args.num_labels_level_2,
            alpha=class_weights_level_2,
            gamma=args.focal_loss_gamma,
            size_average=True
        )
    elif args.use_class_weights:
        loss_fct = nn.CrossEntropyLoss(weight=class_weights_level_2)
    else:
        loss_fct = nn.CrossEntropyLoss()

    loss_level_2 = loss_fct(
        logits_level_2.view(-1, args.num_labels_level_2),
        label_ids_level_2.view(-1)
    )
    return loss_level_2


"""
    optimizer
"""


def get_optimizer1(model, args, learning_rate):
    optimizer_grouped_parameters = []

    # embedding
    if args.model_encoder_type == "albert":
        embedding_params = list(model.albert.embeddings.named_parameters())
    else:
        embedding_params = list(model.bert.embeddings.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters += [
        {
            'params': [p for n, p in embedding_params if not any(nd in n for nd in no_decay)],
            'weigth_decay': args.weight_decay,
            'lr': args.embeddings_learning_rate
        },
        {
            'params': [p for n, p in embedding_params if any(nd in n for nd in no_decay)],
            'weigth_decay': 0.0,
            'lr': args.embeddings_learning_rate
        },
    ]

    # encoder+bert_pooler
    if args.model_encoder_type == "albert":
        encoder_params = list(model.albert.encoder.named_parameters())
        if "bert_pooler" in model.aggregator_names:
            encoder_params = encoder_params + list(model.albert.pooler.named_parameters())

    else:
        encoder_params = list(model.bert.encoder.named_parameters())
        if "bert_pooler" in model.aggregator_names:
            encoder_params = encoder_params + list(model.bert.pooler.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters += [
        {
            'params': [p for n, p in encoder_params if not any(nd in n for nd in no_decay)],
            'weigth_decay': args.weight_decay,
            'lr': args.encoder_learning_rate
        },
        {
            'params': [p for n, p in encoder_params if any(nd in n for nd in no_decay)],
            'weigth_decay': 0.0,
            'lr': args.encoder_learning_rate
        },
    ]

    # linear层 + 初始化的aggregator部分
    classifier_params = list(model.classifier_level_2.named_parameters()) + \
                        list(model.aggregators.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters += [
        {
            'params': [p for n, p in classifier_params if not any(nd in n for nd in no_decay)],
            'weigth_decay': args.weight_decay,
            'lr': args.linear_lr
        },
        {
            'params': [p for n, p in classifier_params if any(nd in n for nd in no_decay)],
            'weigth_decay': 0.0,
            'lr': args.linear_lr
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)

    return optimizer


def get_optimizer(encoder, model, args, learning_rate, remove_pooler=False):
    optimizer_grouped_parameters = []

    # embedding部分
    embeddings_params = list(encoder.embeddings.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters += [
        {'params': [p for n, p in embeddings_params if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay,
         "lr": args.embeddings_learning_rate,
         },
        {'params': [p for n, p in embeddings_params if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         "lr": args.embeddings_learning_rate,
         }
    ]

    # encoder参数
    encoder_params = list(encoder.encoder.named_parameters())
    encoder_params = encoder_params + list(encoder.pooler.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters += [
        {'params': [p for n, p in encoder_params if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay,
         "lr": args.encoder_learning_rate,
         },
        {'params': [p for n, p in encoder_params if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         "lr": args.encoder_learning_rate,
         }
    ]

    # linear层 + 初始化的aggregator部分
    classifier_params = list(model.classifier_level_2.named_parameters()) + \
                        list(model.aggregators.named_parameters())

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters += [
        {'params': [p for n, p in classifier_params if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay,
         "lr": args.linear_lr,
         },
        {'params': [p for n, p in classifier_params if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         "lr": args.linear_lr,
         }
    ]

    # if remove_pooler:
    #     optimizer_grouped_parameters = [n for n in optimizer_grouped_parameters if 'pooler' not in n[0]]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)

    """输出参数数量"""
    total_param1 = 0
    print("MODEL DETAILS:\n")
    for param in encoder.parameters():
        total_param1 += np.prod(list(param.data.size()))
    print('Encoder/Total parameters:', 'Encoder', total_param1)
    total_param2 = 0
    for param in model.parameters():
        total_param2 += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', 'Model', total_param2)
    print('Encoder+MODEL/Total parameters:', 'Encoder+MODEL', total_param1 + total_param2)

    return optimizer


def get_optimizer2(encoder, model, args, learning_rate, remove_pooler=False):
    """
    get BertAdam for encoder / classifier or BertModel
    :param model:
    :param classifier:
    :param args:
    :param remove_pooler:
    :return:
    """

    param_optimizer = list(encoder.named_parameters())
    param_optimizer += list(model.named_parameters())

    total_param1 = 0
    print("MODEL DETAILS:\n")
    for param in encoder.parameters():
        total_param1 += np.prod(list(param.data.size()))
    print('Encoder/Total parameters:', 'Encoder', total_param1)
    total_param2 = 0
    for param in model.parameters():
        total_param2 += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', 'Model', total_param2)
    print('Encoder+MODEL/Total parameters:', 'Encoder+MODEL', total_param1 + total_param2)

    if remove_pooler:
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)

    return optimizer


"""
    count_parameters
"""


def compute_metrics(intent_preds, intent_labels):
    assert len(intent_preds) == len(intent_labels)
    results = {}
    classification_report_dict = classification_report(intent_preds, intent_labels, output_dict=True)
    for key0, val0 in classification_report_dict.items():
        if isinstance(val0, dict):
            for key1, val1 in val0.items():
                results[key0 + "__" + key1] = val1

        else:
            results[key0] = val0
    return results
