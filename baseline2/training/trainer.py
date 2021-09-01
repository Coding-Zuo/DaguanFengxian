import json
import logging
import os
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from tqdm import tqdm, trange
from transformers import AdamW, get_polynomial_decay_schedule_with_warmup, \
    get_cosine_schedule_with_warmup, \
    get_cosine_with_hard_restarts_schedule_with_warmup

from training.configs import MODEL_CLASSES
from training.utils import compute_metrics, get_labels

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args,
                 train_dataset=None,
                 dev_dataset=None,
                 test_dataset=None,
                 train_sample_weights=None,
                 dev_sample_weights=None,
                 test_sample_weights=None,
                 ):
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
        self.label2freq_level_1 = json.load(
            open(args.label2freq_level_1_dir, "r", encoding="utf-8"),

        )
        self.label2freq_level_2 = json.load(
            open(args.label2freq_level_2_dir, "r", encoding="utf-8"),
        )

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.args.device = self.device

        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
        self.config = self.config_class.from_pretrained(args.model_name_or_path,
                                                        finetuning_task=args.task,
                                                        gradient_checkpointing=True)
        self.model = self.model_class.from_pretrained(
            args.model_name_or_path,
            config=self.config,
            args=args,
            label_list_level_1=self.label_list_level_1,
            label_list_level_2=self.label_list_level_2,
            label2freq_level_1=self.label2freq_level_1,
            label2freq_level_2=self.label2freq_level_2,
        )
        self.model.to(self.device)

        # for early  stopping
        self.metric_key_for_early_stop = args.metric_key_for_early_stop
        self.best_score = -1e+10
        self.patience = args.patience
        self.early_stopping_counter = 0
        self.do_early_stop = False

    def train(self):
        if self.args.use_weighted_sampler:
            train_sampler = WeightedRandomSampler(
                self.train_sample_weights,
                len(self.train_sample_weights),
            )
        else:
            train_sampler = RandomSampler(self.train_dataset)

        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.args.train_batch_size
        )

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (
                    len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        for n, p in self.model.named_parameters():
            print(n)

        optimizer_grouped_parameters = []
        # embedding部分
        if self.args.model_type == "albert":
            embeddings_params = list(self.model.albert.embeddings.named_parameters())
        else:
            embeddings_params = list(self.model.bert.embeddings.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters += [
            {'params': [p for n, p in embeddings_params if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay,
             "lr": self.args.embeddings_learning_rate,
             },
            {'params': [p for n, p in embeddings_params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,
             "lr": self.args.embeddings_learning_rate,
             }
        ]

        # encoder + bert_pooler 部分
        if self.args.model_type == "albert":
            encoder_params = list(self.model.albert.encoder.named_parameters())
            if "bert_pooler" in self.model.aggregator_names:
                encoder_params = encoder_params + list(self.model.albert.pooler.named_parameters())

        else:
            encoder_params = list(self.model.bert.encoder.named_parameters())
            if "bert_pooler" in self.model.aggregator_names:
                encoder_params = encoder_params + list(self.model.bert.pooler.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters += [
            {'params': [p for n, p in encoder_params if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay,
             "lr": self.args.encoder_learning_rate,
             },
            {'params': [p for n, p in encoder_params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,
             "lr": self.args.encoder_learning_rate,
             }
        ]

        # linear层 + 初始化的aggregator部分
        classifier_params = list(self.model.classifier_level_2.named_parameters()) + \
                            list(self.model.aggregators.named_parameters())

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters += [
            {'params': [p for n, p in classifier_params if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay,
             "lr": self.args.classifier_learning_rate,
             },
            {'params': [p for n, p in classifier_params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,
             "lr": self.args.classifier_learning_rate,
             }
        ]

        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.args.learning_rate,
                          eps=self.args.adam_epsilon)
        # scheduler = get_linear_schedule_with_warmup(
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=t_total,
        )
        # scheduler = get_polynomial_decay_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=self.args.warmup_steps,
        #     num_training_steps=t_total,
        #     power=2
        # )

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

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'label_ids_level_1': batch[3],
                          'label_ids_level_2': batch[4],
                          }
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                loss = outputs[0]

                del batch

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        results = self.evaluate("dev")

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

                    # if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    #     self.save_model()

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
                epoch_iterator.close()
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
                outputs = self.model(**inputs)
                tmp_eval_loss, logits_level_2 = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
                del batch

            nb_eval_steps += 1

            # label prediction
            if preds_level_2 is None:
                preds_level_2 = logits_level_2.detach().cpu().numpy()
                out_label_ids_level_2 = inputs['label_ids_level_2'].detach().cpu().numpy()
            else:
                preds_level_2 = np.append(preds_level_2, logits_level_2.detach().cpu().numpy(), axis=0)
                out_label_ids_level_2 = np.append(
                    out_label_ids_level_2, inputs['label_ids_level_2'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
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
            f_out = open(os.path.join(self.args.model_dir, "test_predictions.csv"), "w", encoding="utf-8")
            f_out.write("id,label" + "\n")

            list_preds_level_2 = preds_level_2.tolist()
            for i, pred_label_id in enumerate(list_preds_level_2):
                pred_label_name_level_2 = self.label_list_level_2[pred_label_id]
                f_out.write("%s,%s" % (str(i), str(pred_label_name_level_2)) + "\n")

        return results

    def save_model(self):
        # Save models checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained models
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving models checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether models exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(self.args.model_dir,
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
