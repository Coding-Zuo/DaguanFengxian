# -*- coding:utf-8 -*-
"""
    IMPORTING LIBS
"""
import numpy as np
import pandas as pd
import os
import logging
import socket
import time
import random
import glob
import argparse, json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader
from os.path import join
from tensorboardX import SummaryWriter

from tqdm import tqdm
from tqdm import trange
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
import torchsnooper

"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from args_config import get_params
from models.model_envs import MODEL_CLASSES
from dataload.data_loader_bert import load_and_cache_examples
from models.model_envs import load_encoder_model
from models.modeling import DaguanModel
from training.train_eval_optim import get_optimizer1, get_optimizer, get_class_weigth, compute_loss, eval_model

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

#########################################################################
# Initialize arguments
##########################################################################
gpu, config, general, params, gt_params, encoder_params, MODEL_NAME, out_dir = get_params()
logger.info('-' * 100)
logger.info('Input Argument Information')
logger.info('-' * 100)

all_args_dict = {**gpu, **config, **general, **params, **params['schedule'], **params['lr'], **params['loss'],
                 **gt_params, **encoder_params, "out_dir": out_dir, "model_name": MODEL_NAME}

for a in all_args_dict:
    if not isinstance(all_args_dict[a], dict):
        logger.info('%-28s  %s' % (a, all_args_dict[a]))

args = argparse.Namespace(**all_args_dict)

# save args
os.makedirs(out_dir, exist_ok=True)
args_file = "GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y') + "_" + "training_args.bin"
torch.save(all_args_dict, os.path.join(out_dir, args_file))

#########################################################################
# Read Data
##########################################################################
tokenizer = MODEL_CLASSES[args.model_encoder_type][2].from_pretrained(args.encoder_name_or_path)

train_dataset, train_sample_weights = load_and_cache_examples(args, tokenizer, mode="train")
dev_dataset, dev_sample_weights = load_and_cache_examples(args, tokenizer, mode="dev")
test_dataset, test_sample_weights = load_and_cache_examples(args, tokenizer, mode="test")

if args.use_weighted_sampler:
    train_sampler = WeightedRandomSampler(train_sample_weights, len(train_sample_weights))
else:
    train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

eval_sampler = SequentialSampler(dev_dataset)
eval_dataloader = DataLoader(dev_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)

label_list_level_1, label_list_level_2, class_weights_level_1, class_weights_level_2 = get_class_weigth(args)

#########################################################################
# Initialize Model
##########################################################################
args.exp_name = args.out_dir
cached_config_file = join(args.exp_name, 'cached_config.bin1')
if os.path.exists(cached_config_file):
    cached_config = torch.load(cached_config_file)
    encoder_path = join(args.exp_name, cached_config['encoder'])
    model_path = join(args.exp_name, cached_config['model'])
    learning_rate = cached_config['lr']
    start_epoch = cached_config['epoch']
    best_score = cached_config['best_score']
    logger.info("Loading encoder from: {}".format(encoder_path))
    logger.info("Loading model from: {}".format(model_path))
else:
    encoder_path = None
    model_path = None
    start_epoch = 0
    best_score = 0
    learning_rate = args.encoder_learning_rate

device = config['device']

encoder, _ = load_encoder_model(args, args.encoder_name_or_path, args.model_encoder_type)
model = DaguanModel(config=args)

if encoder_path is not None:
    encoder.load_state_dict(torch.load(encoder_path))
if model_path is not None:
    model.load_state_dict(torch.load(model_path))

encoder.to(args.device)
model.to(args.device)

#########################################################################
# Evalaute if resumed from other checkpoint
##########################################################################
# if encoder_path is not None and model_path is not None:
#     output_pred_file = os.path.join(args.exp_name, 'prev_checkpoint.pred.json')
#     output_eval_file = os.path.join(args.exp_name, 'prev_checkpoint.eval.txt')
#     prev_metrics, prev_threshold = eval_model(args, encoder, model,
#                                               dev_dataloader, dev_example_dict, dev_feature_dict,
#                                               output_pred_file, output_eval_file, DEV_GOLD_FILE)
#     logger.info("Best threshold for prev checkpoint: {}".format(prev_threshold))
#     for key, val in prev_metrics.items():
#         logger.info("{} = {}".format(key, val))


#########################################################################
# Get Optimizer
##########################################################################
t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

optimizer = get_optimizer(encoder, model, args, args.encoder_learning_rate, remove_pooler=False)

if args.fp16:
    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    models, optimizer = amp.initialize([encoder, model], optimizer, opt_level=args.fp16_opt_level)
    assert len(models) == 2
    encoder, model = models

# Distributed training (should be after apex fp16 initialization)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

#########################################################################
# launch training
##########################################################################
logger.info("***** Running training *****")
logger.info("  Num examples = %d", len(train_dataset))
logger.info("  Num Epochs = %d", args.num_train_epochs)
logger.info("  Total train batch size = %d", args.batch_size)
logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
logger.info("  Total optimization steps = %d", t_total)
logger.info("  Logging steps = %d", args.logging_steps)
logger.info("  Save steps = %d", args.save_steps)

global_step = 0
tr_loss, logging_loss = 0.0, 0.0
early_stopping_counter = 0
do_early_stop = False

if args.local_rank in [-1, 0]:
    tb_writer = SummaryWriter(args.exp_name)

encoder.zero_grad()
model.zero_grad()

train_iterator = trange(start_epoch, start_epoch + int(args.num_train_epochs), desc="Epoch",
                        disable=args.local_rank not in [-1, 0])

for epoch in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

    for step, batch in enumerate(epoch_iterator):
        encoder.train()
        model.train()

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2], }  # XLM don't use segment_ids

        outputs = encoder(**inputs)
        batch.append(outputs[0])
        batch.append(outputs[1])

        outputs = model(batch)

        loss = compute_loss(args, class_weights_level_1, class_weights_level_2, outputs[0], batch[4])
        del batch

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        # with torchsnooper.snoop():
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        tr_loss += loss.item()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            encoder.zero_grad()
            model.zero_grad()
            global_step += 1

            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                results = eval_model(args, encoder,
                                     model, eval_dataloader, "dev", label_list_level_2, class_weights_level_1,
                                     class_weights_level_2)

                logger.info("*" * 50)
                logger.info("current step score for metric_key_for_early_stop: {}".format(
                    results.get(args.metric_key_for_early_stop, 0.0)))
                logger.info("best score for metric_key_for_early_stop: {}".format(best_score))
                logger.info("*" * 50)

                avg_loss = (tr_loss - logging_loss) / (args.logging_steps * args.gradient_accumulation_steps)

                loss_str = "step[{0:6}] loss[{1:.5f}]"
                logger.info(loss_str.format(global_step, avg_loss))

                # tensorboard
                tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                tb_writer.add_scalar('loss',
                                     (tr_loss - logging_loss) / (args.logging_steps * args.gradient_accumulation_steps),
                                     global_step)
                logging_loss = tr_loss

                if results.get(args.metric_key_for_early_stop, ) > best_score:
                    best_score = results.get(args.metric_key_for_early_stop, )
                    early_stopping_counter = 0
                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        torch.save({'epoch': epoch + 1,
                                    'lr': scheduler.get_last_lr()[0],
                                    'encoder': 'encoder.pkl',
                                    'model': 'model.pkl',
                                    'best_score': best_score, },
                                   join(args.exp_name, f'cached_config.bin')
                                   )
                        torch.save({k: v.cpu() for k, v in encoder.state_dict().items()},
                                   join(args.exp_name, f'encoder_{epoch + 1}.pkl'))
                        torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                                   join(args.exp_name, f'model_{epoch + 1}.pkl'))
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= args.patience:
                        do_early_stop = True
                        logger.info("best score is {}".format(best_score))

                if do_early_stop:
                    break

        if 0 < args.max_steps < global_step:
            epoch_iterator.close()
            break

        if do_early_stop:
            epoch_iterator.close()
            break

    if 0 < args.max_steps < global_step:
        train_iterator.close()
        break

    if do_early_stop:
        epoch_iterator.close()
        break

    print('final:', tr_loss / global_step)

    results = eval_model(args, encoder,
                         model, eval_dataloader, "test", label_list_level_2, class_weights_level_1,
                         class_weights_level_2)
