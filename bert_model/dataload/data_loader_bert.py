# -*- coding:utf-8 -*-
import os
import copy
import json
import logging
import math
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
    def __init__(self, input_ids, attention_mask, token_type_ids, label_id_level_1, label_id_level_2, id):
        self.input_ids = input_ids
        self.id = id  ######################
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
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

    def __init__(self, args):
        self.args = args
        self.labels_level_1 = get_labels(args.label_file_level_1)
        self.labels_level_2 = get_labels(args.label_file_level_2)

    def get_examples(self, mode):
        data_path = os.path.join(self.args.data_dir, "{}.txt".format(mode))
        logger.info("LOOKING AT {}".format(data_path))
        if self.args.vocab_mapping:
            return self._create_examples(lines=self._read_file(data_path), set_type=mode, sep="\t")
        else:
            return self._create_examples(lines=self._read_file(data_path), set_type=mode)

    @classmethod
    def _read_file(cls, input_file, skip_first_line=False):
        with open(input_file, 'r', encoding="utf-8") as f:
            lines = []
            for i, line in enumerate(f):
                if skip_first_line:
                    if i == 0: continue

                lines.append(line.strip())
            return lines

    def _create_examples(self, lines, set_type, sep=","):
        examples = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line: continue

            line = line.split(sep)

            # id
            if set_type == "dev":  ######################
                id_ = line[3]
            else:
                id_ = line[0]
            guid = "%s-%s" % (set_type, id_)

            # input_text
            words = line[1].split()
            words = [w.strip() for w in words if len(w.strip()) > 0]

            # label
            if set_type == "test":
                label_level_1 = 0
                label_level_2 = 0
            else:
                # if not len(line) == 4:
                #     print(line)
                # if not len(line) == 3:
                #     print(line)

                label_name = line[2]
                label_name_level_1 = label_name.split("-")[0]
                label_name_level_2 = label_name

                label_level_1 = self.labels_level_1.index(label_name_level_1)
                label_level_2 = self.labels_level_2.index(label_name_level_2)

            # if set_type == "dev":
            #     guid = line[-1]
            examples.append(InputExample(
                guid=guid,
                words=words,
                label_level_1=label_level_1,
                label_level_2=label_level_2
            ))
        return examples


processors = {
    "daguan": DaguanDataProcessor,
}


def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True,
                                 label2freq_level_2=None,
                                 label_list_level_2=None):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    sample_weights = []
    for (ex_index, example) in enumerate(examples):
        tokens = tokenizer.tokenize(" ".join(example.words))

        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]

        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids),
                                                                                                  max_seq_len)

        label_id_level_1 = int(example.label_level_1)
        label_id_level_2 = int(example.label_level_2)

        sample_weight = math.sqrt(1 / label2freq_level_2[label_list_level_2[label_id_level_2]])
        sample_weights.append(sample_weight)

        # if ex_index < 10:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % example.guid)
        #     logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        #     logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
        #     logger.info("label_level_1: %s (id = %d)" % (example.label_level_1, label_id_level_1))
        #     logger.info("label_level_2: %s (id = %d)" % (example.label_level_2, label_id_level_2))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label_id_level_1=label_id_level_1,
                label_id_level_2=label_id_level_2,
                id=int(example.guid.split("-")[1]),  ######################
            )
        )

    return features, sample_weights


def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.task](args)

    # level 标签频次
    label2freq_level_1 = json.load(open(args.label2freq_level_1, 'r', encoding="utf-8"))
    label2freq_level_2 = json.load(open(args.label2freq_level_2, 'r', encoding='utf-8'))

    # 加载label list
    label_list_level_1 = get_labels(args.label_file_level_1)
    label_list_level_2 = get_labels(args.label_file_level_2)

    args.num_labels_level_1 = len(label_list_level_1)
    args.num_labels_level_2 = len(label_list_level_2)

    # load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        mode, args.task, args.model_type, args.max_seq_len))
    cached_sampling_weights_file = os.path.join(args.data_dir, 'cached_sampling_weights_{}_{}_{}_{}'.format(
        mode, args.task, args.model_type, args.max_seq_len))

    if os.path.exists(cached_features_file) and os.path.exists(cached_sampling_weights_file):  ######################
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        sampling_weights = torch.load(cached_sampling_weights_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        features, sampling_weights = convert_examples_to_features(examples, args.max_seq_len,
                                                                  tokenizer, label2freq_level_2=label2freq_level_2,
                                                                  label_list_level_2=label_list_level_2)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
        logger.info("Saving features into cached file %s", cached_sampling_weights_file)
        torch.save(sampling_weights, cached_sampling_weights_file)

    # convert to tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_id_level_1s = torch.tensor([f.label_id_level_1 for f in features], dtype=torch.long)
    all_label_id_level_2s = torch.tensor([f.label_id_level_2 for f in features], dtype=torch.long)
    all_id = torch.tensor([f.id for f in features], dtype=torch.long)  ######################

    dataset = TensorDataset(
        all_input_ids.to(args.device),
        all_attention_mask.to(args.device),
        all_token_type_ids.to(args.device),
        all_label_id_level_1s.to(args.device),
        all_label_id_level_2s.to(args.device),
        all_id.to(args.device),  ######################
    )

    return dataset, sampling_weights
