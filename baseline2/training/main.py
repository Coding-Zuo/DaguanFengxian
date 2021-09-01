# -*- coding: utf-8 -*-

import argparse

import sys
import os

sys.path.insert(0, "./")
sys.path.append("./")
from training.configs import MODEL_CLASSES, MODEL_PATH_MAP

from training.trainer import Trainer
from training.utils import init_logger, set_seed
from training.data_loader import load_and_cache_examples

import warnings

warnings.filterwarnings("ignore")
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # 按照PCI_BUS_ID顺序从0开始排列GPU设备
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def main(args):
    init_logger()
    set_seed(args)
    tokenizer = MODEL_CLASSES[args.model_type][2].from_pretrained(
        args.model_name_or_path
    )

    tmp_res = tokenizer.tokenize("[MASK]")
    print("tmp_res: ", tmp_res)

    train_dataset, train_sample_weights = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset, dev_sample_weights = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset, test_sample_weights = load_and_cache_examples(args, tokenizer, mode="test")

    print("train_dataset: ", len(train_dataset))
    print("train_sample_weights: ", len(train_sample_weights))
    print("dev_dataset: ", len(dev_dataset))
    print("dev_sample_weights: ", len(dev_sample_weights))
    print("test_dataset: ", len(test_dataset))
    print("test_sample_weights: ", len(test_sample_weights))

    trainer = Trainer(
        args,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        train_sample_weights=train_sample_weights,
        dev_sample_weights=dev_sample_weights,
        test_sample_weights=test_sample_weights,
    )

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("dev")

        trainer.evaluate("test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load models")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--label_file_level_1", default=None, type=str, help="Label file for level 1 label")
    parser.add_argument("--label_file_level_2", default=None, type=str, help="Label file for level 2 label")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=50, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    # ----------------------------------------------------------------------

    parser.add_argument("--embeddings_learning_rate", default=1e-4,
                        type=float, help="The learning rate for Adam.")
    parser.add_argument("--encoder_learning_rate", default=5e-4,
                        type=float, help="The learning rate for Adam.")
    parser.add_argument("--classifier_learning_rate", default=5e-4,
                        type=float, help="The learning rate for Adam.")

    # 是否使用 use_lstm
    parser.add_argument("--use_lstm", action="store_true",
                        help="Whether to use lstm.")

    parser.add_argument("--patience", default=6, type=int,
                        help="patience for early stopping ")
    parser.add_argument("--metric_key_for_early_stop", default="macro avg__f1-score", type=str,
                        help="metric name for early stopping ")

    # 可以选择用不同的aggregator
    parser.add_argument("--aggregator_names", default="bert_pool", type=str,
                        help="Model type selected in the list: [bert_pooler, slf_attn_pool, max_pool, avg_pool, dr_pool, ] ")

    # 针对不均衡样本
    parser.add_argument("--use_focal_loss", action="store_true",
                        help="use focal loss")
    parser.add_argument(
        "--focal_loss_gamma", default=2.0, type=float,
        help="gamma in focal loss"
    )
    # use_class_weights
    parser.add_argument("--use_class_weights", action="store_true",
                        help="whether to use class weights; ")

    # 是否使用weighted sampler
    parser.add_argument("--use_weighted_sampler", action="store_true",
                        help="use weighted sampler")
    parser.add_argument(
        "--label2freq_level_1_dir", default=None, type=str,
        help="path to the level 1 label2freq dict;"
    )
    parser.add_argument(
        "--label2freq_level_2_dir", default=None, type=str,
        help="path to the level 2 label2freq dict;"
    )

    parser.add_argument(
        "--model_name_or_path", default=None, type=str,
        help="path to the pretrained lm;"
    )

    # ----------------------------------------------------------------------

    args = parser.parse_args()

    if not args.model_name_or_path:
        args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    main(args)
