# -*- coding:utf-8 -*-
import os
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


def get_labels(label_file):
    return [label.strip() for label in open(label_file, 'r', encoding='utf-8')]


class InputExample(object):
    def __init__(self, guid, words, label_level_1=None, label_level_2=None):
        self.guid = guid
        self.words = words
        self.label_level_1 = label_level_1
        self.label_level_2 = label_level_2

    def __repr__(self):
        return str(self.to_json_string())

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output


class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, label_id_level_1, label_id_level_2):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label_id_level_1 = label_id_level_1
        self.label_id_level_2 = label_id_level_2

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DaguanDataProcessor(object):
    """Processor for the BERT dataload set """

    def __init__(self, args):
        self.args = args
        self.labels_level_1 = get_labels(args.label_file_level_1)
        self.labels_level_2 = get_labels(args.label_file_level_2)

    @classmethod
    def _read_file(self, input_file, skip_first_line=False):
        """Reads a tab separated value file."""
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            for i, line in enumerate(f):
                if skip_first_line:
                    if i == 0:
                        continue
                lines.append(line.strip())
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets"""
        examples = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            line = line.split(",")
            # id
            id_ = line[0]
            guid = "%s-%s" % (set_type, id_)

            # 1.input_text
            words = line[1].split()
            words = [w.strip() for w in words if len(w.strip()) > 0]

            # 标签
            if set_type == "test":
                label_level_1 = 0
                label_level_2 = 0
            else:
                label_name = line[2]
                label_name_level_1 = label_name.split("-")[0]
                label_name_level_2 = label_name

                label_level_1 = self.labels_level_1.index(label_name_level_1)
                label_level_2 = self.labels_level_2.index(label_name_level_2)

            examples.append(
                InputExample(
                    guid=guid,
                    words=words,
                    label_level_1=label_level_1,
                    label_level_2=label_level_2
                )
            )
        return examples

    def get_examples(self, mode):
        data_path = os.path.join(self.args.data_dir, "{}.txt".format(mode))
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(lines=self._read_file(data_path), set_type=mode)


processors = {
    "daguan": DaguanDataProcessor
}


def convert_examples_to_features(examples,
                                 max_seq_len,
                                 pad_token_id=0,
                                 unk_token_id=1,
                                 mask_padding_with_zero=True,
                                 vocab_list=None):
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # tokenizer word by word (for NER)
        tokens = example.words

        # Account for [CLS] and [SEP]
        special_tokens_count = 0
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]

        input_ids = [vocab_list.index(w) if w in vocab_list else unk_token_id for w in tokens]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len)

        label_id_level_1 = int(example.label_level_1)
        label_id_level_2 = int(example.label_level_2)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("label_level_1: %s (id = %d)" % (example.label_level_1, label_id_level_1))
            logger.info("label_level_2: %s (id = %d)" % (example.label_level_2, label_id_level_2))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          label_id_level_1=label_id_level_1,
                          label_id_level_2=label_id_level_2
                          ))

    return features


def load_and_cache_examples(args, mode, vocab_list=None):
    processor = processors[args.task](args)

    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}'.format(mode, args.task, args.max_seq_len))

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == 'train':
            examples = processor.get_examples("train")
        elif mode == 'dev':
            examples = processor.get_examples("dev")
        elif mode == 'test':
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        features = convert_examples_to_features(examples, args.max_seq_len, vocab_list=vocab_list)

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

        # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_label_id_level_1s = torch.tensor([f.label_id_level_1 for f in features], dtype=torch.long)
    all_label_id_level_2s = torch.tensor([f.label_id_level_2 for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids,
                            all_attention_mask,
                            all_label_id_level_1s,
                            all_label_id_level_2s,
                            )

    return dataset
