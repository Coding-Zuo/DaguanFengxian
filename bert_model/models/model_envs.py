import os
import logging
import sys
from os.path import join, dirname, abspath
from transformers import (BertConfig, BertTokenizer, RobertaConfig, RobertaTokenizer, AlbertConfig, AlbertTokenizer,
                          AlbertModel)
from transformers import (BertModel, RobertaModel)


# Add submodule path into import paths
# is there a better way to handle the sub module path append problem?
# PROJECT_FOLDER = os.path.dirname(__file__)
# PROJECT_FOLDER = dirname(dirname(abspath(__file__)))
# sys.path.append(join(PROJECT_FOLDER, 'transformers'))


def get_cached_filename_v1(f_type, config):
    assert f_type in ['examples', 'features', 'graphs']
    return f"cached_{f_type}_{config.model_type}_{config.max_seq_length}_{config.max_query_length}.pkl.gz"


############################################################
# Model Related Global Varialbes
############################################################

def load_encoder_model(args, encoder_name_or_path, model_type):
    if encoder_name_or_path in [None, 'None', 'none']:
        raise ValueError('no checkpoint provided for model!')

    config_class, model_encoder, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(encoder_name_or_path)
    args.embed_dim = config.hidden_size
    if config is None:
        raise ValueError(f'config.json is not found at {encoder_name_or_path}')

    if os.path.exists(encoder_name_or_path):
        if os.path.isfile(os.path.join(encoder_name_or_path, 'pytorch_model.bin')):
            encoder_file = os.path.join(encoder_name_or_path, 'pytorch_model.bin')
        else:
            encoder_file = os.path.join(encoder_name_or_path, 'encoder.pkl')
        # encoder = os.path.join(encoder_file, 'encoder.pkl')
        encoder = model_encoder.from_pretrained(encoder_file, config=config)
    else:
        encoder = model_encoder.from_pretrained(encoder_name_or_path, config=config)
    return encoder, config


# ALL_MODELS = sum(
#     (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig, AlbertConfig)), ())

from models.modeling_bert import ClsBERT
from models.modeling_albert import ClsALBERT
from models.modeling_nezha1 import ClsNezha
from models.modeling_bert_pabee import ClsBERTWithPABEE
from models.modeling_nezha_pabee import ClsNezhaWithPABEE

MODEL_CLASSES = {
    'bert': (BertConfig, ClsBERT, BertTokenizer),
    'bert_pabee': (BertConfig, ClsBERTWithPABEE, BertTokenizer),
    'nezha': (BertConfig, ClsNezha, BertTokenizer),
    'nezha_pabee': (BertConfig, ClsNezhaWithPABEE, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'albert': (AlbertConfig, ClsALBERT, BertTokenizer),
    'macbert': (BertConfig, ClsBERT, BertTokenizer),
}
