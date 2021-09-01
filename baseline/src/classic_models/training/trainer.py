# -*- coding:utf-8 -*-
import os
import logging
import random

import time

from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from sklearn.metrics import classification_report

from src.classic_models.models.modeling import ClsModel

logger = logging.getLogger(__name__)


def get_labels(label_file):
    return [label.strip() for label in open(label_file, 'r', encoding='utf-8')]


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


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.label_list_level_1 = get_labels(args.label_file_level_1)
        self.label_list_level_2 = get_labels(args.label_file_level_2)

        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.args.device = self.device

        self.model = ClsModel(args, self.label_list_level_1, self.label_list_level_2).to(self.device)

        # for early stopping
        self.metric_key_for_early_stop = args.metric_key_for_early_stop
        self.best_score = -1e+10
        self.patience = args.patience
        self.early_stopping_counter = 0
        self.do_early_stop = False

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (
                    len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        for n, p in self.model.named_parameters():
            print(n)
        optimizer_grouped_parameters = []

        # embedding部分
        embeddings_params = list(self.model.embeddings.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        if not self.args.embeddings_learning_rate:
            self.args.embeddings_learning_rate = self.args.learning_rate
        optimizer_grouped_parameters += [
            {
                'params': [p for n, p in embeddings_params if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay,
                'lr': self.args.embeddings_learning_rate
            },
            {
                'params': [p for n, p in embeddings_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.args.embeddings_learning_rate
            }
        ]

        # encoder+aggregator
        encoder_params = list(self.model.encoder.named_parameters()) + list(self.model.aggregator.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters += [
            {'params': [p for n, p in encoder_params if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay,
             "lr": self.args.learning_rate,
             },
            {'params': [p for n, p in encoder_params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,
             "lr": self.args.learning_rate,
             }
        ]

        # linear层
        classifier_params = list(self.model.classifier.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        if not self.args.classifier_learning_rate:
            self.args.classifier_learning_rate = self.args.learning_rate
        optimizer_grouped_parameters += [
            {'params': [p for n, p in classifier_params if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay,
             "lr": self.args.classifier_learning_rate,
             },
            {'params': [p for n, p in classifier_params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,
             "lr": self.args.classifier_learning_rate,
             },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                              num_training_steps=t_total, power=2)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc='Epoch')

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc='Iteration')
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)

                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'label_ids_level_1': batch[2],
                    'label_ids_level_2': batch[3]
                }
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        results = self.evaluate('dev')

                        logger.info("*" * 50)
                        logger.info("current step score for metric_key_for_early_stop: {}".format(
                            results.get(self.metric_key_for_early_stop, 0.0)))
                        logger.info("best score for metric_key_for_early_stop: {}".format(self.best_score))
                        logger.info("*" * 50)

                        if results.get(self.metric_key_for_early_stop, ) > self.best_score:
                            self.best_score = results.get(self.metric_key_for_early_stop, )
                            self.early_stopping_counter = 0
                            if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                                self.save_model()
                        else:
                            self.early_stopping_counter += 1
                            if self.early_stopping_counter >= self.patience:
                                self.do_early_stop = True
                                logger.info("best score is {}".format(self.best_score))

                        if self.do_early_stop:
                            break
                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

                if self.do_early_stop:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

            if self.do_early_stop:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds_level_1 = None
        preds_level_2 = None
        out_label_ids_level_1 = None
        out_label_ids_level_2 = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc='Evaluting'):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'label_ids_level_1': batch[2],
                    'label_ids_level_2': batch[3],
                }
                outputs = self.model(**inputs)
                tmp_eval_loss, logits_level2 = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds_level_2 is None:
                preds_level_2 = logits_level2.detach().cpu().numpy()
                out_label_ids_level_2 = inputs['label_ids_level_2'].detach().cpu().numpy()
            else:
                preds_level_2 = np.append(preds_level_2, logits_level2.detach().cpu().numpy(), axis=0)
                out_label_ids_level_2 = np.append(
                    out_label_ids_level_2, inputs['label_ids_level_2'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {'loss': eval_loss}

        # label prediction result
        preds_level_2 = np.argmax(preds_level_2, axis=1)
        results_level_2 = compute_metrics(preds_level_2, out_label_ids_level_2)
        for key_, val_ in results_level_2.items():
            results[key_ + "__level_2"] = val_

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        if mode == 'test':
            f_out = open(os.path.join(self.args.model_dir, 'test_predictions.csv'), 'w', encoding='utf-8')
            f_out.write('id,label' + '\n')

            list_preds_level_2 = preds_level_2.tolist()
            for i, pred_label_id in enumerate(list_preds_level_2):
                pred_label_name_level_2 = self.label_list_level_2[pred_label_id]
                f_out.write("%s,%s" % (str(i), str(pred_label_name_level_2)) + "\n")

        return results

    def save_model(self):
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        model_to_save_dir = os.path.join(self.args.model_dir, 'pytorch_model.bin')
        torch.save(model_to_save.state_dict(), model_to_save_dir)
        logger.info("Model weights saved in {}".format(model_to_save_dir))

        # Save training arguments together with the trained models
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving models checkpoint to %s", self.args.model_dir)

    def load_model(self):
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            model_to_load_dir = os.path.join(self.args.model_dir, "pytorch_model.bin")
            model_state_dict = torch.load(model_to_load_dir)
            self.model.load_state_dict(model_state_dict)

            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some models files might be missing...")
