# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn as nn

from models.model_utils import get_embedding_matrix_and_vocab, replace_masked_values
from models.model_utils import get_sinusoid_encoding_table, masked_softmax, weighted_sum


#########################################################################
# Classifier
##########################################################################
class Classifier(nn.Module):
    def __init__(self, args, input_dim=128, num_labels=2):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.linear = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


#########################################################################
# Aggregator
##########################################################################
class AggregatorLayer(nn.Module):
    def __init__(self, args, aggregator_name=None):
        super(AggregatorLayer, self).__init__()
        self.args = args
        self.d_model = args.hidden_dim
        self.aggregator_op_name = aggregator_name

        self.aggregator_op = None
        if self.aggregator_op_name == 'slf_attn_pool':
            attn_vector = nn.Linear(self.d_model, 1)
            self.aggregator_op = SelfAttnAggregator(self.d_model, attn_vector=attn_vector)
        elif self.aggregator_op_name == 'max_pool':
            self.aggregator_op = MaxPoolerAggregator()
        else:
            self.aggregator_op = AvgPoolerAggregator()

    def forward(self, input_tensors, mask=None):
        output = self.aggregator_op(input_tensors, mask)
        return output


class SelfAttnAggregator(nn.Module):
    def __init__(self, output_dim, attn_vector=None):
        super(SelfAttnAggregator, self).__init__()
        self.output_dim = output_dim
        self.attn_vector = None
        if attn_vector:
            self.attn_vector = attn_vector
        else:
            self.attn_vector = nn.Linear(self.output_dim, 1)

    def forward(self, input_tensors, mask):
        self_attentive_logits = self.attn_vector(input_tensors).squeeze(2)
        self_weights = masked_softmax(self_attentive_logits, mask)
        input_self_attn_pooled = weighted_sum(input_tensors, self_weights)
        return input_self_attn_pooled


class MaxPoolerAggregator(nn.Module):
    def __init__(self):
        super(MaxPoolerAggregator, self).__init__()

    def forward(self, input_tensors, mask):
        if mask is not None:
            input_tensors = replace_masked_values(input_tensors, mask.unsqueeze(2), -1e7)
        input_max_pooled = torch.max(input_tensors, 1)[0]
        return input_max_pooled


class AvgPoolerAggregator(nn.Module):
    def __init__(self, ) -> None:
        super(AvgPoolerAggregator, self).__init__()

    def forward(self, input_tensors, mask):
        if mask is not None:
            input_tensors = replace_masked_values(input_tensors, mask.unsqueeze(2), 0)
        tokens_avg_pooled = torch.mean(input_tensors, 1)
        return tokens_avg_pooled


#########################################################################
# Encoder
##########################################################################
class TextCnnEncoder(nn.Module):
    def __init__(self, args):
        super(TextCnnEncoder, self).__init__()
        self.args = args

        # 4个卷积核
        self.ops = nn.ModuleList()
        for kernel_size in [1, 3, 5, 7]:
            op_ = ChildSepConv(self.args.embed_dim, self.args.hidden_dim, kernel_size)
            self.ops.append(op_)

        self.dropout = nn.Dropout(p=args.dropout_rate)
        self.LayerNorm = nn.LayerNorm(self.args.hidden_dim)

    def forward(self, input_tensors=None, attention_mask=None, position_ids=None, **kwargs):
        tmp_outputs = []
        for i, op in enumerate(self.ops):
            input_tensors_conv = op(input_tensors)
            tmp_outputs.append(input_tensors_conv)

        output_tensors = sum(tmp_outputs)
        output_tensors = self.dropout(output_tensors)
        output_tensors = self.LayerNorm(output_tensors)
        return output_tensors


class BiLSTMEncoder(nn.Module):
    def __init__(self, args):
        super(BiLSTMEncoder, self).__init__()
        self.args = args

        self.op = RnnEncoder(
            self.args.embed_dim,
            self.args.hidden_dim,
            rnn_name="lstm",
            bidirectional=True
        )
        self.dropout = nn.Dropout(p=args.dropout)
        self.LayerNorm = nn.LayerNorm(self.args.hidden_dim)

    def forward(self, input_tensors=None, attention_mask=None, position_ids=None, **kwargs):
        output_tensors = self.op(input_tensors)
        output_tensors = self.dropout(output_tensors)
        output_tensors = self.LayerNorm(output_tensors)
        return output_tensors


class RnnEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, rnn_name, bidirectional):
        super(RnnEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rnn_name = rnn_name
        self.bidirecitonal = bidirectional

        if bidirectional:
            assert output_dim % 2 == 0
            hidden_size = output_dim // 2
        else:
            hidden_size = output_dim

        if rnn_name == 'lstm':
            self._rnn = torch.nn.LSTM(input_dim, hidden_size, num_layers=1, batch_first=True,
                                      bidirectional=bidirectional, bias=False)
        else:
            self._rnn = torch.nn.GRU(input_dim, hidden_size, num_layers=1, batch_first=True,
                                     bidirectional=bidirectional, bias=False)

    def forward(self, input_tensors, mask=None):
        encoded_output, _ = self._rnn(input_tensors)
        return encoded_output


class ChildSepConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size

        self.deptwis_conv = nn.Conv1d(
            in_channels=int(in_ch),
            out_channels=int(in_ch),
            groups=int(in_ch),
            kernel_size=int(kernel_size),
            padding=int(kernel_size // 2),
            bias=False
        )

        self.pointwise_conv = nn.Conv1d(
            in_channels=int(in_ch),
            out_channels=int(out_ch),
            kernel_size=1,
            padding=0,
            bias=False
        )

        self.op = nn.Sequential(self.deptwis_conv, nn.ReLU(inplace=False), self.pointwise_conv)

    def forward(self, x, mask=None):
        x = x.transpose(1, 2)
        # x.size
        x_conv = self.op(x)
        #
        x_conv = x_conv.transpose(1, 2)
        #
        if self.kernel_size % 2 == 0:
            x_conv = x_conv[:, :, :-1]

        return x_conv


class EmbeddingLayer(nn.Module):
    def __init__(self, args):
        super(EmbeddingLayer, self).__init__()
        self.args = args

        self.embed_dim = args.embed_dim

        # 加载embedding：从word2vec
        vocab_list, vector_list = get_embedding_matrix_and_vocab(args.w2v_file, skip_first_line=True)
        self.vocab_list = vocab_list

        assert self.embed_dim == len(vector_list[0])
        assert len(vocab_list) == len(vector_list)

        self.w2v_matrix = np.asarray(vector_list)

        # 初始化embedding
        if args.random_init_w2v:
            self.word_embedding = nn.Embedding(len(self.vocab_list), self.embed_dim)
        else:
            self.word_embedding = nn.Embedding(
                len(self.vocab_list),
                self.embed_dim
            ).from_pretrained(torch.FloatTensor(self.w2v_matrix), freeze=False)

        self.positional_embedding = SinusordPositionEmbedding(
            max_len=args.max_seq_len,
            embed_dim=self.embed_dim
        )

        self.dropout = nn.Dropout(p=args.dropout_rate)
        self.LayerNorm = nn.LayerNorm(self.embed_dim)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        device = input_ids.device

        input_embeds = self.word_embedding(input_ids)

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        position_embeddings = self.positional_embedding(position_ids)

        embeddings = input_embeds + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SinusordPositionEmbedding(nn.Module):
    def __init__(self, max_len=512, embed_dim=300) -> None:
        super(SinusordPositionEmbedding, self).__init__()
        self.encoder_position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(max_len, embed_dim, padding_idx=0),
            freeze=True
        )

    def forward(self, input_pos_tensors: torch.Tensor):
        return self.encoder_position_enc(input_pos_tensors)
