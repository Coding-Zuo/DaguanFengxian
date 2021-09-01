# -*- coding:utf-8 -*-
import argparse
import os
import torch
import json
import logging
import time
import random
import numpy as np

logger = logging.getLogger(__name__)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


"""
    GPU Setup
"""


def gpu_setup(use_gpu, gpu_id, local_rank, data_parallel, config):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # 按照PCI_BUS_ID顺序从0开始排列GPU设备
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        if local_rank == -1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if data_parallel:
                config['gpu']['n_gpu'] = len(gpu_id.split(','))
            else:
                config['gpu']['n_gpu'] = 1
        else:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            torch.distributed.init_process_group(backend="nccl")
            config['gpu']['n_gpu'] = 1
    else:
        print('cuda not available')
        device = torch.device("cpu")

    config['device'] = device
    return device


"""
python src/bert_models/training/main.py --model_type bert 
--model_name_or_path resources/bert/bert-base-chinese_embedding_replaced_random 
--data_dir ./datasets/phase_1/splits/fold_0 --label_file_level_1 datasets/phase_1/labels_level_1.txt 
--label_file_level_2 datasets/phase_1/labels_level_2.txt --task daguan --aggregator bert_pooler 
--model_dir ./experiments/outputs/daguan/bert_0821_0 --do_train --do_eval --train_batch_size 16 
--num_train_epochs 50 --embeddings_learning_rate 0.4e-4 --encoder_learning_rate 0.5e-4 
--classifier_learning_rate 5e-4 --warmup_steps 200 --max_seq_len 132 --dropout_rate 0.15 
--metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 200 --patience 12 
--label2freq_level_1_dir datasets/phase_1/label2freq_level_1.json 
--label2freq_level_2_dir datasets/phase_1/label2freq_level_2.json 
"""


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True,
                        help="Please give a config.json file with training/model/data/param details")
    # gpu
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--local_rank', help="For distributed training: local_rank")
    parser.add_argument('--data_parallel', help="For distributed training: data_parallel")
    # general
    parser.add_argument('--task', help="Please give a value for task name")
    parser.add_argument('--do_train', help="Please give a value for do_train")
    parser.add_argument('--do_eval', help="Please give a value for do_eval")
    parser.add_argument('--fp16', help="Please give a value for fp16")
    parser.add_argument('--fp16_opt_level', help="Please give a value for fp16_opt_level")
    parser.add_argument('--vocab_mapping', help="Please give a value for vocab_mapping")
    parser.add_argument('--main_data_dir', help="Please give a value for main_data_dir")
    parser.add_argument('--data_dir', help="Please give a value for data_dir")
    parser.add_argument('--label_file_level_1', help="Please give a value for label_level_1")
    parser.add_argument('--label_file_level_2', help="Please give a value for label_level_2")
    parser.add_argument('--label2freq_level_1', help="Please give a value for label2freq_level_1_dir")
    parser.add_argument('--label2freq_level_2', help="Please give a value for label2freq_level_2_dir")
    parser.add_argument('--exp_name', help="Please give a value for exp_name")
    parser.add_argument('--model_type', help="Please give a value for model_type name")
    parser.add_argument('--model_encoder_type', help="Please give a value for model_encoder_type")
    parser.add_argument('--encoder_name_or_path', help="Please give a value for encoder_name_or_path")
    # params
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--metric_key_for_early_stop', help="Please give a value for metric_key_for_early_stop")
    parser.add_argument('--logging_steps', help="Please give a value for logging_steps")
    parser.add_argument('--save_steps', help="Please give a value for save_steps")
    parser.add_argument('--patience', help="Please give a value for patience")
    parser.add_argument('--max_steps',
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")
    # # lr
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--linear_lr', help="Please give a value for linear_lr")
    parser.add_argument('--encoder_learning_rate', help="Please give a value for encoder_learning_rate")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--embeddings_learning_rate', help="Please give a value for embeddings_learning_rate")
    # # schedule
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--warmup_steps', help="Please give a value for warmup_steps")
    parser.add_argument('--max_grad_norm', help="Please give a value for max_grad_norm")
    parser.add_argument('--num_train_epochs', help="Please give a value for num_train_epochs")
    parser.add_argument('--per_gpu_train_batch_size', help="Please give a value for per_gpu_train_batch_size")
    parser.add_argument('--gradient_accumulation_steps', help="Please give a value for gradient_accumulation_steps")
    parser.add_argument('--eval_batch_size', help="Please give a value for eval_batch_size")
    # # loss
    parser.add_argument("--loss_fct_name", default=True, help="main loss function: "
                                                              "(1) 'ce', cross entropy loss; "
                                                              "(2) 'focal', focal loss; "
                                                              "(3) 'dice', dice loss;")
    parser.add_argument("--contrastive_loss", default=True, help="which contrastive loss to use: "
                                                                 "(1) 'ntxent_loss';"
                                                                 "(2) 'supconloss';")
    parser.add_argument("--what_to_contrast", default=True, help="what to contrast in each batch: "
                                                                 "(1) 'sample';"
                                                                 "(2) 'sample_and_class_embeddings';")
    parser.add_argument("--contrastive_loss_weight", default=True, help="loss weight for ntxent")
    parser.add_argument("--contrastive_temperature", default=True, help="temperature for contrastive loss")
    parser.add_argument("--use_focal_loss", default=True)
    parser.add_argument("--focal_loss_gamma", type=float, default=2.0)
    parser.add_argument("--use_class_weights", default=True)
    parser.add_argument("--use_weighted_sampler", default=True)
    # gt_params
    parser.add_argument('--use_lstm', help="Please give a value for use_lstm")
    parser.add_argument('--aggregator', help="Please give a value for aggregator_names")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--use_fgm', help="Please give a value for use_fgm")
    parser.add_argument('--use_pgd', help="Please give a value for use_pgd")
    # encoder_params
    parser.add_argument('--max_seq_len', help="Please give a value for max_seq_length")
    parser.add_argument('--do_lower_case', help="Please give a value for do_lower_case")
    ##############################################

    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    # device
    gpu = config['gpu']
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
        config['gpu']['local_rank'] = int(args.local_rank)
        config['gpu']['data_parallel'] = True if args.data_parallel == 'True' else False

    # general
    general = config['general']
    if args.do_train is not None:
        general['do_train'] = True if args.do_train == 'True' else False
    if args.do_eval is not None:
        general['do_eval'] = True if args.do_eval == 'True' else False
    if args.fp16 is not None:
        general['fp16'] = True if args.fp16 == 'True' else False
    if args.fp16_opt_level is not None:
        general['fp16_opt_level'] = args.fp16_opt_level
    if args.vocab_mapping is not None:
        general['vocab_mapping'] = True if args.vocab_mapping == 'True' else False
    if args.main_data_dir is not None:
        main_data_dir = args.main_data_dir
    else:
        main_data_dir = config['general']['main_data_dir']
    if args.exp_name is not None:
        out_dir = main_data_dir + "outputs/" + args.exp_name
    else:
        out_dir = main_data_dir + "outputs/" + config['general']['exp_name']
    if args.data_dir is not None:
        general['data_dir'] = main_data_dir + args.data_dir
    else:
        general['data_dir'] = main_data_dir + config['general']['data_dir']
    if args.model_type is not None:
        MODEL_NAME = args.model_type
    else:
        MODEL_NAME = config['general']['model_type']
    if args.task is not None:
        general['task'] = args.task
    else:
        general['task'] = config['general']['task']
    if args.model_encoder_type is not None:
        general['model_encoder_type'] = main_data_dir + args.model_encoder_type
    if args.label_file_level_1 is not None:
        general['label_file_level_1'] = main_data_dir + args.label_file_level_1
    else:
        general['label_file_level_1'] = main_data_dir + config['general']['label_file_level_1']
    if args.label_file_level_2 is not None:
        general['label_file_level_2'] = main_data_dir + args.label_file_level_2
    else:
        general['label_file_level_2'] = main_data_dir + config['general']['label_file_level_2']
    if args.label2freq_level_1 is not None:
        general['label2freq_level_1'] = main_data_dir + args.label2freq_level_1
    else:
        general['label2freq_level_1'] = main_data_dir + config['general']['label2freq_level_1']
    if args.label2freq_level_2 is not None:
        general['label2freq_level_2'] = main_data_dir + args.label2freq_level_2
    else:
        general['label2freq_level_2'] = main_data_dir + config['general']['label2freq_level_2']

    # params
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.metric_key_for_early_stop is not None:
        params['metric_key_for_early_stop'] = args.metric_key_for_early_stop
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    if args.logging_steps is not None:
        params['logging_steps'] = float(args.logging_steps)
    if args.save_steps is not None:
        params['save_steps'] = float(args.save_steps)
    if args.max_steps is not None:
        params['max_steps'] = float(args.max_steps)
    if args.patience is not None:
        params['patience'] = float(args.patience)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = float(args.print_epoch_interval)
    # # lr
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.linear_lr is not None:
        params['linear_lr'] = float(args.linear_lr)
    if args.encoder_learning_rate is not None:
        params['encoder_learning_rate'] = float(args.encoder_learning_rate)
    if args.embeddings_learning_rate is not None:
        params['embeddings_learning_rate'] = float(args.embeddings_learning_rate)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    # # schedule
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.warmup_steps is not None:
        params['warmup_steps'] = int(args.warmup_steps)
    if args.max_grad_norm is not None:
        params['max_grad_norm'] = float(args.max_grad_norm)
    if args.num_train_epochs is not None:
        params['num_train_epochs'] = int(args.num_train_epochs)
    if args.per_gpu_train_batch_size is not None:
        params['per_gpu_train_batch_size'] = int(args.per_gpu_train_batch_size)
    if args.gradient_accumulation_steps is not None:
        params['gradient_accumulation_steps'] = int(args.gradient_accumulation_steps)
    if args.eval_batch_size is not None:
        params['eval_batch_size'] = int(args.eval_batch_size)
    # # loss
    if args.use_focal_loss is not None:
        params['use_focal_loss'] = True if args.use_focal_loss == 'True' else False
    if args.focal_loss_gamma is not None:
        params['focal_loss_gamma'] = float(args.focal_loss_gamma)
    if args.loss_fct_name is not None:
        params['loss_fct_name'] = float(args.loss_fct_name)
    if args.use_class_weights is not None:
        params['use_class_weights'] = True if args.use_class_weights == 'True' else False
    if args.use_weighted_sampler is not None:
        params['use_weighted_sampler'] = True if args.use_weighted_sampler == 'True' else False

    # net parameters
    net_params = config['net_params']
    if args.use_lstm is not None:
        net_params['use_lstm'] = True if args.use_lstm == 'True' else False
    if args.aggregator is not None:
        net_params['aggregator'] = args.aggregator
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = args.hidden_dim
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.use_fgm is not None:
        net_params['use_fgm'] = True if args.use_fgm == 'True' else False
    if args.use_pgd is not None:
        net_params['use_pgd'] = True if args.use_pgd == 'True' else False

    encoder_params = config['encoder_params']
    if args.max_seq_len is not None:
        encoder_params['max_seq_len'] = int(args.max_seq_len)
    if args.do_lower_case is not None:
        encoder_params['do_lower_case'] = int(args.do_lower_case)

    gpu_setup(config['gpu']['use'], config['gpu']['id'], config['gpu']['local_rank'],
              config['gpu']['data_parallel'], config)

    params['batch_size'] = params['schedule']['per_gpu_train_batch_size'] * max(1, config['gpu']['n_gpu'])

    general['input_dim'] = 768 if 'base' in general['model_encoder_type'] else (
        4096 if 'albert' in general['model_encoder_type'] else 1024)

    seed_everything(params['seed'])

    return gpu, config, general, params, net_params, encoder_params, MODEL_NAME, out_dir
