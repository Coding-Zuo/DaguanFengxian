# -*- coding:utf-8 -*-
import argparse
import sys
import random
import logging
import torch
import numpy as np

from models.model_utils import get_embedding_matrix_and_vocab
from src.classic_models.training.trainer import Trainer
from src.classic_models.training.data_loader import load_and_cache_examples

sys.path.append('./')


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def main(args):
    init_logger()
    set_seed(args)

    # load vocab and w2v
    vocab_list, vector_list = get_embedding_matrix_and_vocab(args.w2v_file, skip_first_line=True)

    train_dataset = load_and_cache_examples(args, mode='train', vocab_list=vocab_list)
    dev_dataset = load_and_cache_examples(args, mode='dev', vocab_list=vocab_list)
    test_dataset = load_and_cache_examples(args, mode='test', vocab_list=vocab_list)

    print("train_dataset: ", len(train_dataset))
    print("dev_dataset: ", len(dev_dataset))
    print("test_dataset: ", len(test_dataset))

    trainer = Trainer(
        args, train_dataset, dev_dataset, test_dataset
    )

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("dev")

        trainer.evaluate("test")


"""
python src/classic_models/training/main.py --data_dir ./datasets/phase_1/splits/fold_0 
--label_file_level_1 datasets/phase_1/labels_level_1.txt 
--label_file_level_2 datasets/phase_1/labels_level_2.txt 
--task daguan --random_init_w2v --encoder lstm --aggregator max_pool 
--model_dir ./experiments/outputs/daguan/lstm_0815_1 --do_train 
--do_eval --train_batch_size 32 --num_train_epochs 50 
--embeddings_learning_rate 6e-4 --learning_rate 20e-4 
--classifier_learning_rate 20e-4 --warmup_steps 200 --max_seq_len 128 
--hidden_dim 256 --embed_dim 256 
--w2v_file resources/word2vec/dim_256/w2v.vectors 
--dropout_rate 0.2 --metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 400 --patience 5 
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="daguan", type=str, help="The name of the task to train")
    parser.add_argument("--model_dir",
                        default="/data2/code/DaguanFengxian/baseline/experiments/outputs/lstm_max_pool_wv256_epoch100", type=str,
                        help="Path to save, load models")
    parser.add_argument("--data_dir", default="/data2/nlpData/daguanfengxian/phase_1/splits/fold_0", type=str,
                        help="The input dataload dir")
    parser.add_argument("--label_file_level_1", default="/data2/nlpData/daguanfengxian/phase_1/labels_level_1.txt",
                        type=str, help="Label file for level 1 label")
    parser.add_argument("--label_file_level_2", default="/data2/nlpData/daguanfengxian/phase_1/labels_level_2.txt",
                        type=str, help="Label file for level 2 label")

    parser.add_argument('--seed', type=int, default=41, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=128, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--num_train_epochs", default=100.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=200, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=400, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    # ----------------------------------------------------------------------
    # embedding: 随机初始化；
    parser.add_argument("--random_init_w2v", action="store_true",
                        help="是否直接随机初始化embedding； ")

    parser.add_argument("--encoder", default="lstm", type=str,
                        help="Model type selected in the list: [textcnn, lstm] ")
    parser.add_argument("--aggregator", default="max_pool", type=str,
                        help="Model type selected in the list: [slf_attn_pool, max_pool, avg_pool, dr_pool, ] ")

    parser.add_argument("--embed_dim", default=256,
                        type=int, help="dims for embedding layer.")
    parser.add_argument("--hidden_dim", default=256,
                        type=int, help="dims for intermediate layers.")

    parser.add_argument("--embeddings_learning_rate", default=6e-4,
                        type=float, help="The learning rate for Adam.")
    parser.add_argument("--learning_rate", default=20e-4,
                        type=float, help="The learning rate for Adam.")
    parser.add_argument("--classifier_learning_rate", default=20e-4,
                        type=float, help="The learning rate for Adam.")

    parser.add_argument("--w2v_file",
                        default="/data2/nlpData/daguanfengxian/word2vec/dim_256_sg_0_hs_1_epochs_30/w2v.vectors",
                        type=str,
                        help="path to pretrained word2vec file")

    parser.add_argument("--dropout_rate", default=0.2, type=float,
                        help="dropout_rate ")

    parser.add_argument("--patience", default=5, type=int,
                        help="patience for early stopping ")
    parser.add_argument("--metric_key_for_early_stop", default="macro avg__f1-score__level_2", type=str,
                        help="metric name for early stopping ")

    # prediction_output_file
    parser.add_argument("--prediction_output_file", default=None, type=str,
                        help="file for writing out the predictions ")

    # 针对不均衡样本
    parser.add_argument(
        "--class_weights_level_1", default=None, type=str,
        help="class_weights, written in string like '1.0,2.0,2.0,5.0,200.0,300.0,400.0,500.0,500.0' "
    )
    #
    parser.add_argument(
        "--class_weights_level_2", default=None, type=str,
        help="class_weights, written in string like '0.828,1.241,1.465,1.622,1.963,2.002,2.173,2.507,2.564,2.572,2.707,4.244,4.469,4.953,5.460,5.693,6.477,6.694,7.174,7.804,8.648,8.988,9.090,10.06,10.25,11.94,15.53,25.80,32.65,48.48,50.0,80.0,84.21,88.88,100.0' "
    )

    parser.add_argument("--use_focal_loss", default="True", help="use focal loss")
    parser.add_argument(
        "--focal_loss_gamma", default=2.0, type=float,
        help="gamma in focal loss"
    )

    # ----------------------------------------------------------------------

    args = parser.parse_args()

    main(args)
