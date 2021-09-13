# coding=utf-8
# Copyright 2018 The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert BERT checkpoint."""

from __future__ import print_function

import argparse
import os
import re

import numpy as np
import tensorflow as tf
import torch

from transformers import BertConfig, BertForPreTraining


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path, ):
    config_path = os.path.abspath(bert_config_file)
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {} with config at {}".format(tf_path, config_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    for name, array in zip(names, arrays):
        name = name.split('/')
        if name[0] in ['global_step', "bad_steps", "good_steps", "lamb_m", "lamb_v", "loss_scale"]:
            continue
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "lamb_m", "lamb_v"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name[-13:] == '_embeddings_2':
            pointer = getattr(pointer, 'weight')
            array = np.transpose(array)
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf_checkpoint_path", default="/data2/pre-model/nezha/NEZHA-Large-WWM/model.ckpt", type=str,
                        help="Path to tf checkpoint")
    parser.add_argument("--bert_config_file", default="/data2/pre-model/nezha/NEZHA-Large-WWM/bert_config.json",
                        type=str, help="The config file")
    parser.add_argument("--pytorch_dump_path", default="/data2/pre-model/nezha/NEZHA-Large-WWM/pytorch_model.bin",
                        type=str,
                        help="Path to the output pytorch model")

    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path,
                                     args.bert_config_file,
                                     args.pytorch_dump_path, )

    # python src/bert_models/model_process/nezha_convert_tf_checkpoint_to_pytorch.py --tf_checkpoint_path resources/nezha/NEZHA-Base/model.ckpt-900000 --bert_config_file resources/nezha/NEZHA-Base/bert_config.json --pytorch_dump_path resources/nezha/NEZHA-Base/pytorch_model.bin
    # python src/bert_models/model_process/nezha_convert_tf_checkpoint_to_pytorch.py --tf_checkpoint_path resources/nezha/NEZHA-Large/model.ckpt-325810 --bert_config_file resources/nezha/NEZHA-Large/bert_config.json --pytorch_dump_path resources/nezha/NEZHA-Large/pytorch_model.bin
