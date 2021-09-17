# -*- coding:utf-8 -*-
import json
import logging
import os
import time
import warnings

import xgboost as xgb
import lightgbm as lgb
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from tqdm import tqdm, trange
from sklearn.metrics import classification_report
from transformers import AdamW, get_polynomial_decay_schedule_with_warmup, \
    get_cosine_schedule_with_warmup, \
    get_cosine_with_hard_restarts_schedule_with_warmup
from torchcontrib.optim import SWA
from collections import defaultdict

from dataload.data_loader_bert import get_labels
from training.train_eval_optim import get_optimizer1, compute_metrics
from models.model_envs import MODEL_CLASSES

from training.Adversarial import FGM, PGD, getDelta, updateDelta, FreeLB, FGM1, PGD1, FGM2, PGD2
import random

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None, train_sample_weights=None,
                 dev_sample_weights=None, test_sample_weights=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        # for weighted sampling
        self.train_sample_weights = train_sample_weights
        self.dev_sample_weights = dev_sample_weights
        self.test_sample_weights = test_sample_weights

        self.label_list_level_1 = get_labels(args.label_file_level_1)
        self.label_list_level_2 = get_labels(args.label_file_level_2)

        # level 标签的频次
        self.label2freq_level_1 = json.load(open(args.label2freq_level_1, 'r', encoding='utf-8'))
        self.label2freq_level_2 = json.load(open(args.label2freq_level_2, 'r', encoding='utf-8'))

        self.device = args.device

        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_encoder_type]
        self.config = self.config_class.from_pretrained(args.encoder_name_or_path,
                                                        finetuning_task=args.task,
                                                        gradient_checkpointing=True)
        self.model = self.model_class.from_pretrained(
            args.encoder_name_or_path,
            args=args,
            label_list_level_1=self.label_list_level_1,
            label_list_level_2=self.label_list_level_2,
            label2freq_level_1=self.label2freq_level_1,
            label2freq_level_2=self.label2freq_level_2,
        ).to(self.device)

        # for early stopping
        self.metric_key_for_early_stop = args.metric_key_for_early_stop
        self.best_score = -1e+10
        self.patience = args.patience
        self.early_stopping_counter = 0
        self.do_early_stop = False

        # Adv
        self.adv_trainer = None
        if self.args.use_fgm and self.args.use_pgd:
            raise Exception("Adv methed !!!")
        if self.args.use_fgm:
            self.adv_trainer = FGM2(self.model, epsilon=self.args.epsilon_for_adv,
                                    emb_names=self.args.emb_names.split(","))
        if self.args.use_pgd:
            self.adv_trainer = PGD2(self.model, epsilon=self.args.epsilon_for_adv,
                                    alpha=self.args.alpha_for_adv, emb_names=self.args.emb_names.split(","))

    def train(self):
        if self.args.use_weighted_sampler:
            train_sampler = WeightedRandomSampler(self.train_sample_weights, len(self.train_sample_weights))
        else:
            train_sampler = RandomSampler(self.train_dataset)

        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (
                    len(train_dataloader) // self.args.gradient_accumulation_steps)
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        optimizer = get_optimizer1(self.model, self.args, self.args.lr)
        # if self.args.use_swa:
        #     optimizer = SWA(optimizer, swa_start=0, swa_freq=10, swa_lr=7e-5)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=t_total)

        swa_model = torch.optim.swa_utils.AveragedModel(self.model)
        swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=0.05)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        train_loss_all = []
        dev_loss_all = []
        swa_flag = False
        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)

                # if self.args.use_freelb:
                #     embeds_init = self.model.bert.embeddings.word_embeddings(batch[0])
                #     d = getDelta(attention_mask=batch[1], embeds_init=embeds_init)
                #     d.requires_grad_()
                #     embeds_init = embeds_init + d

                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'label_ids_level_1': batch[3],
                    'label_ids_level_2': batch[4]
                }
                if self.args.model_encoder_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]

                if self.args.use_freelb:
                    # if False:
                    loss, _ = self.freelb.attack(self.model, inputs)
                else:
                    outputs = self.model(**inputs)
                    loss = outputs[0]
                    # 洪泛法
                    if self.args.use_hongfan:
                        b = 0.2
                        loss = (loss - b).abs() + b
                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps

                    # with torch.cuda.amp.scale_loss(loss, optimizer) as scaled_loss:
                    #     scaled_loss.backward()
                    loss.backward()

                # 对抗训练
                if self.adv_trainer is not None:
                    if random.uniform(0, 1) <= self.args.adv_rate:  # 随机攻击
                        if self.args.use_fgm:
                            self.adv_trainer.backup_grad()
                            # 实施对抗
                            self.adv_trainer.attack()
                            # 梯度清零，用于计算在对抗样本处的梯度
                            self.model.zero_grad()
                            outputs_adv = self.model(**inputs)
                            loss_adv = outputs_adv[0]
                            loss_adv.backward()
                            # embedding(被攻击的模块)的梯度回复原值，其他部分梯度累加，
                            # 这样相当于综合了两步优化的方向
                            self.adv_trainer.restore_grad()
                            # 恢复embedding的参数
                            self.adv_trainer.restore()
                        elif self.args.use_pgd:
                            self.adv_trainer.backup_grad()  # 保存正常的grad
                            # 对抗训练
                            for t in range(self.args.steps_for_adv):
                                self.adv_trainer.attack(is_first_attack=(t == 0))
                                self.model.zero_grad()
                                outputs_adv = self.model(**inputs)
                                loss_adv = outputs_adv[0]
                                loss_adv.backward()
                            self.adv_trainer.restore_grad()
                            self.adv_trainer.restore()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    # if self.args.use_swa and (epoch > 3 or swa_flag):
                    #     optimizer.update_swa()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        results = self.evaluate('dev')
                        train_loss_all.append(loss.item())
                        dev_loss_all.append(results['loss'])

                        logger.info("*" * 50)
                        logger.info("current step score for metric_key_for_early_stop: {}".format(
                            results.get(self.metric_key_for_early_stop, 0.0)))
                        logger.info("best score for metric_key_for_early_stop: {}".format(self.best_score))
                        logger.info("*" * 50)

                        if results.get(self.metric_key_for_early_stop, ) > self.best_score:
                            self.best_score = results.get(self.metric_key_for_early_stop, )
                            self.early_stopping_counter = 0
                            self.save_model()
                        else:
                            self.early_stopping_counter += 1
                            if self.early_stopping_counter > self.patience:
                                self.do_early_stop = True

                                logger.info("best score is {}".format(self.best_score))

                        if self.do_early_stop:
                            # if self.args.use_swa:
                            #     # 训练结束时使用收集到的swa moving average
                            #     optimizer.swap_swa_sgd()
                            break
                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

                if self.do_early_stop:
                    epoch_iterator.close()
                    break

                time.sleep(0.5)

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

            if self.do_early_stop:
                epoch_iterator.close()
                break

        return global_step, tr_loss / global_step, train_loss_all, dev_loss_all

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

        # 存储每层的结果
        layer_idx2preds_level_2 = None
        if "pabee" in self.args.model_encoder_type:
            layer_idx2preds_level_2 = {
                layer_idx: None
                for layer_idx in range(self.config.num_hidden_layers)
            }

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'label_ids_level_1': batch[3],
                    'label_ids_level_2': batch[4],
                }

                if self.args.model_encoder_type != 'distibert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)

                if self.args.use_multi_task:
                    tmp_eval_loss, logits_level_2 = outputs[2], outputs[4]
                else:
                    tmp_eval_loss, logits_level_2 = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # label prediction
            if preds_level_2 is None:
                preds_level_2 = logits_level_2.detach().cpu().numpy()
                out_label_ids_level_2 = inputs['label_ids_level_2'].detach().cpu().numpy()
            else:
                preds_level_2 = np.append(preds_level_2, logits_level_2.detach().cpu().numpy(), axis=0)
                out_label_ids_level_2 = np.append(out_label_ids_level_2,
                                                  inputs['label_ids_level_2'].detach().cpu().numpy(), axis=0)

            # label prediction for each layer
            if "pabee" in self.args.model_encoder_type:
                all_logits_level_2 = outputs[2]
                for i, logits_ix in enumerate(all_logits_level_2):
                    if not isinstance(layer_idx2preds_level_2[i], np.ndarray):
                        layer_idx2preds_level_2[i] = logits_ix.detach().cpu().numpy()
                    else:
                        layer_idx2preds_level_2[i] = np.append(
                            layer_idx2preds_level_2[i],
                            logits_ix.detach().cpu().numpy(),
                            axis=0
                        )

        eval_loss = eval_loss / nb_eval_steps
        results = {
            'loss': eval_loss
        }

        preds_level_2 = np.argmax(preds_level_2, axis=1)
        results_level_2 = compute_metrics(preds_level_2, out_label_ids_level_2)
        for key_, val_ in results_level_2.items():
            results[key_ + "__level_2"] = val_

        ############################
        # 对每层输出一个
        ############################
        if "pabee" in self.args.model_encoder_type:
            for layer_idx in range(self.config.num_hidden_layers):
                preds = layer_idx2preds_level_2[layer_idx]
                # label prediction result layer "layer_ix"
                preds = np.argmax(preds, axis=1)
                results_idx = compute_metrics(preds, out_label_ids_level_2)
                for key_, val_ in results_idx.items():
                    results[key_ + "__level_2" + "__layer_{}".format(layer_idx)] = val_

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        if mode == 'test':
            f_out = open(os.path.join(self.args.exp_name, "test_predictions.csv"), "w", encoding="utf-8")
            f_out.write("id,label" + "\n")

            list_preds_level_2 = preds_level_2.tolist()
            for i, pred_label_id in enumerate(list_preds_level_2):
                pred_label_name_level_2 = self.label_list_level_2[pred_label_id]
                f_out.write("%s,%s" % (str(i), str(pred_label_name_level_2)) + "\n")

        if "pabee" in self.args.model_encoder_type:
            if mode == "test":

                for layer_idx in range(self.config.num_hidden_layers):
                    f_out = open(os.path.join(self.args.exp_name, "test_predictions_layer_{}.csv".format(layer_idx)),
                                 "w", encoding="utf-8")
                    f_out.write("id,label" + "\n")

                    preds = layer_idx2preds_level_2[layer_idx]

                    # label prediction result at layer "layer_idx"
                    preds = np.argmax(preds, axis=1)

                    list_preds = preds.tolist()
                    for i, pred_label_id in enumerate(list_preds):
                        pred_label_name_level_2 = self.label_list_level_2[pred_label_id]
                        f_out.write("%s,%s" % (str(i), str(pred_label_name_level_2)) + "\n")

        return results

    def save_model(self):
        if not os.path.exists(self.args.exp_name):
            os.makedirs(self.args.exp_name)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        if self.args.model_encoder_type == 'nezha' or self.args.model_encoder_type == 'nezha_pabee':
            json.dump(
                model_to_save.config.__dict__,
                open(os.path.join(self.args.exp_name, 'config.json'), 'w', encoding="utf-8"),
                ensure_ascii=False,
                indent=2
            )
            state_dict = model_to_save.state_dict()
            output_model_file = os.path.join(self.args.exp_name, "pytorch_model.bin")
            torch.save(state_dict, output_model_file)
        else:
            model_to_save.save_pretrained(self.args.exp_name)

        torch.save(self.args, os.path.join(self.args.exp_name, 'training_args_bin'))
        logger.info("Saving models checkpoint to %s", self.args.exp_name)

    def load_model(self):
        if not os.path.exists(self.args.exp_name):
            raise Exception("Model doesn't exists! Train first!")

        try:
            if self.args.model_encoder_type == "nezha" or self.args.model_encoder_type == 'nezha_pabee':
                output_model_file = os.path.join(self.args.exp_name, "pytorch_model.bin")
                self.model.load_state_dict(torch.load(output_model_file, map_location=self.device))
            else:
                self.model = self.model_class.from_pretrained(self.args.exp_name,
                                                              config=self.config,
                                                              args=self.args,
                                                              label_list_level_1=self.label_list_level_1,
                                                              label_list_level_2=self.label_list_level_2,
                                                              label2freq_level_1=self.label2freq_level_1,
                                                              label2freq_level_2=self.label2freq_level_2,
                                                              )
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some models files might be missing...")

    def tree_model_cls(self, mode, gb_tree):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        elif mode == 'train':
            dataset = self.train_dataset
            eval_dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        # 顺序采样
        trian_sampler = SequentialSampler(dataset)
        train_dataloader = DataLoader(dataset, sampler=trian_sampler, batch_size=self.args.eval_batch_size)

        # eval不进行梯度更新
        self.model.eval()
        features = []
        for batch in tqdm(train_dataloader, desc="Train eval"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'label_ids_level_1': batch[3],
                          'label_ids_level_2': batch[4],
                          }
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                if len(features) == 0:
                    features = self.model(**inputs)[-1].detach().cpu().numpy()
                    labels = batch[4].detach().cpu().numpy()
                else:
                    features = np.append(features, self.model(**inputs)[-1].detach().cpu().numpy(), axis=0)
                    labels = np.append(labels, batch[4].detach().cpu().numpy(), axis=0)

        # gb分类
        if gb_tree == 'xgb':
            xgb_cls = xgb.XGBClassifier(
                max_depth=6, learning_rate=0.05, n_estimators=100,
                objective="multi:softprob", num_class=35,
                subsample=0.8, colsample_bytree=0.8, tree_method='gpu_hist',
                min_child_samples=3, eval_metric='auc', reg_lambda=0.5
            )
            xgb_cls.fit(features, labels, verbose=True)
            results = xgb_cls.predict(features)
            print(classification_report(labels, results))

            features = []
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = tuple(t.to(self.device) for t in batch)
                with torch.no_grad():
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'label_ids_level_1': batch[3],
                              'label_ids_level_2': batch[4],
                              }
                    if self.args.model_type != 'distilbert':
                        inputs['token_type_ids'] = batch[2]
                    # print(len(features))
                    if len(features) == 0:
                        features = self.model(**inputs)[-1].detach().cpu().numpy()
                        labels = batch[4].detach().cpu().numpy()
                        # print(features.shape)
                    else:
                        features = np.append(features, self.model(**inputs).detach().cpu().numpy(), axis=0)
                        labels = np.append(labels, batch[4].detach().cpu().numpy(), axis=0)
            results = xgb_cls.predict(features)
            print(classification_report(labels, results))

    def ensemble_test(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        eval_loss = 0.0
        nb_eval_steps = 0
        preds_level_1 = None
        preds_level_2 = None
        preds_level_2_prob = []
        out_label_ids_level_1 = None
        out_label_ids_level_2 = None
        id_prob_dict = defaultdict()

        self.model.eval()

        i = 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)

            # i += 1
            # if i == 30:
            #     break

            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'label_ids_level_1': batch[3],
                    'label_ids_level_2': batch[4],
                }

                if self.args.model_encoder_type != 'distibert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)

                if self.args.use_multi_task:
                    tmp_eval_loss, logits_level_2 = outputs[2], outputs[4]
                else:
                    tmp_eval_loss, logits_level_2 = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # label prediction
            if preds_level_2 is None:
                torch.set_printoptions(profile="full")
                preds_level_2 = logits_level_2.detach().cpu().numpy()
                out_label_ids_level_2 = inputs['label_ids_level_2'].detach().cpu().numpy()
            else:
                preds_level_2 = np.append(preds_level_2, logits_level_2.detach().cpu().numpy(), axis=0)
                out_label_ids_level_2 = np.append(out_label_ids_level_2,
                                                  inputs['label_ids_level_2'].detach().cpu().numpy(), axis=0)
            preds_level_2_prob.extend(torch.nn.Softmax(dim=1)(logits_level_2).detach().cpu().numpy())
            for ids, prob in zip(batch[5], preds_level_2_prob):
                id_ = ids.item()
                id_prob_dict[id_] = prob

        eval_loss = eval_loss / nb_eval_steps
        results = {
            'loss': eval_loss
        }
        max_logits = np.max(preds_level_2, axis=1).tolist()
        preds_level_2_max = np.argmax(preds_level_2, axis=1)

        list_preds_level_2 = preds_level_2_max.tolist()

        max_logits_dict = {}
        for idx, (logit, label) in enumerate(zip(max_logits, list_preds_level_2)):
            max_logits_dict[idx] = (logit, label)

        return list_preds_level_2, max_logits_dict, preds_level_2_prob, id_prob_dict, preds_level_2
