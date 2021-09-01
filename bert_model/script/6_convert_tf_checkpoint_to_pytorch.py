# -*- coding:utf-8 -*-
import argparse
import logging
import torch
from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert

logging.basicConfig(level=logging.INFO)

# https://blog.csdn.net/weixin_41287060/article/details/105080705
# https://github.com/huggingface/transformers/issues/393
def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    '''
        tf_checkpoint_path: ckpt文件
        bert_config_file: json文件
        pytorch_dump_path: pytorch模型保存位置
    '''

    # 初始化pytorch模型
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    # 从checkpoint中加载权重
    load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    # 保存pytorch模型
    print("Save Pytorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf_checkpoint_path", default="/data2/pre-model/nezha/NEZHA-Base-WWM/model.ckpt", type=str, help="Path to tf checkpoint")
    parser.add_argument("--bert_config_file", default="/data2/pre-model/nezha/NEZHA-Base-WWM/bert_config.json", type=str, help="The config file")
    parser.add_argument("--pytorch_dump_path", default="/data2/pre-model/nezha/NEZHA-Base-WWM/pytorch_model.bin", type=str,
                        help="Path to the output pytorch model")

    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.bert_config_file, args.pytorch_dump_path)
