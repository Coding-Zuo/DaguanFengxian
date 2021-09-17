# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import get_embedding_matrix_and_vocab, replace_masked_values
from models.model_utils import get_sinusoid_encoding_table, masked_softmax, weighted_sum
from torch.nn.utils import rnn
from torch.autograd import Variable


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


class MultiSampleClassifier(nn.Module):
    def __init__(self, args, input_dim, num_labels=2):
        super(MultiSampleClassifier, self).__init__()
        self.args = args
        self.linear = nn.Linear(input_dim, num_labels)
        self.dropout_ops = nn.ModuleList([
            nn.Dropout(args.dropout) for _ in range(self.args.dropout_num)
        ])

    def forward(self, x):
        logits = None
        for i, dropout_op in enumerate(self.dropout_ops):
            if i == 0:
                out = dropout_op(x)
                logits = self.linear(out)
            else:
                temp_out = dropout_op(x)
                temp_logits = self.linear(temp_out)
                logits += temp_logits
        # 相加还是取平均
        if self.args.ms_average:
            logits = logits / self.args.dropout_num
        return logits


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
        elif self.aggregator_op_name == 'dr_pooler':
            cap_num_ = 4  # 胶囊数
            iter_num_ = 3  # 迭代次数
            shared_fc_ = nn.Linear(self.d_model, self.d_model)
            self.aggregator_op = DynamicRoutingAggregator(
                input_dim=self.d_model,
                out_caps_num=cap_num_,
                out_caps_dim=int(self.d_model / cap_num_),
                iter_num=iter_num_,
                shared_fc=shared_fc_,
                device=args.device
            )
        else:
            self.aggregator_op = AvgPoolerAggregator()

    def forward(self, input_tensors, mask=None):
        output = self.aggregator_op(input_tensors, mask)
        return output


class DynamicRoutingAggregator(nn.Module):
    def __init__(self, input_dim, out_caps_num, out_caps_dim, iter_num, device,
                 output_format='flatten', activation_function='tanh', shared_fc=None):
        super(DynamicRoutingAggregator, self).__init__()
        self.input_dim = input_dim
        self.out_cpas_num = out_caps_num
        self.out_caps_dim = out_caps_dim
        self.iter_num = iter_num
        self.output_format = output_format
        self.activation_function = activation_function
        self.device = device
        if shared_fc:
            self.shared_fc = shared_fc
        else:
            self.shared_fc = nn.Linear(input_dim, out_caps_dim * out_caps_num)

    def squash(self, input_tensors, dim=2):
        norm = torch.norm(input_tensors, 2, dim=dim, keepdim=True)  # [batch_size, out_caps_num, 1]
        norm_sq = norm ** 2
        s = norm_sq / (1.0 + norm_sq) * input_tensors / torch.sqrt(norm_sq + 1e-8)
        return s

    def forward(self, input_tensors, mask):
        """
        input_tensors : (batch_size, num_tokens, input_dim).
        mask : sentence mask, (batch_size, num_tokens).
        output_tensors : torch.FloatTensor
            if "flatten":
                return tensor of shape ``(batch_size, out_caps_num * out_caps_dim)`` .
            else:
                return tensor of shape ``(batch_size, out_caps_num, out_caps_dim)``
        """
        # shared caps
        batch_size = input_tensors.size()[0]
        num_tokens = input_tensors.size()[1]

        shared_info = self.shared_fc(input_tensors)  # (bs, num_tokens, out_caps_dim * out_caps_num)
        if self.activation_function == 'tanh':
            shared_info = torch.tan(shared_info)
        elif self.activation_function == 'relu':
            shared_info = F.relu(shared_info)

        shared_info = shared_info.view(-1, num_tokens, self.out_cpas_num, self.out_caps_dim)

        assert len(mask.size()) == 2
        mask_float = torch.unsqueeze(mask, dim=-1).to(torch.float32)  # [bs, seq_len ,1]

        B = torch.zeros([batch_size, num_tokens, self.out_cpas_num], dtype=torch.float32).to(self.device)

        # [bs, seq_len ,1] -> [bs, seq_len, out_caps_num]
        mask_tiled = mask.unsqueeze(-1).repeat(1, 1, self.out_cpas_num)

        B = B.masked_fill((1 - mask_tiled).byte(), -1e32)

        for i in range(self.iter_num):
            C = F.softmax(B, dim=2)
            C = C * mask_float  # (bs,num_tokens, out_caps_num)
            C = torch.unsqueeze(C, dim=-1)  # (bs,num_tokens,out_caps_num,1)

            weighted_uhat = C * shared_info  # (bs, out_caps_num,out_caps_dim)
            S = torch.sum(weighted_uhat, dim=1)  # [batch_size, out_caps_num, out_caps_dim]

            V = self.squash(S, dim=1)  # [batch_size, out_caps_num, out_caps_dim]
            V = torch.unsqueeze(V, dim=1)  # [batch_size, 1, out_caps_num, out_caps_dim]

            B += torch.sum((shared_info * V).detach(), dim=-1)  # [batch_size, num_tokens, out_caps_num]

        V_ret = torch.squeeze(V, dim=1)  # (batch_size, out_caps_num, out_caps_dim)

        if self.output_format == 'flatten':
            V_ret = V_ret.view([-1, self.out_cpas_num * self.out_caps_dim])

        return V_ret


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
        # input:[bs, seq_len, hidden_dim] output:[bs,output_dim]
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
            self.args.hidden_dim,
            self.args.hidden_dim,
            rnn_name="lstm",
            bidirectional=True
        )
        self.dropout = nn.Dropout(p=args.dropout)
        self.LayerNorm = nn.LayerNorm(self.args.hidden_dim)

    def forward(self, input_tensors=None, attention_mask=None, position_ids=None, **kwargs):
        input_tensors = self.LayerNorm(input_tensors)
        output_tensors = self.op(input_tensors)
        output_tensors = self.dropout(output_tensors)
        return output_tensors


class BiGRUEncoder(nn.Module):
    def __init__(self, args):
        super(BiGRUEncoder, self).__init__()
        self.args = args

        self.op = RnnEncoder(
            self.args.hidden_dim,
            self.args.hidden_dim,
            rnn_name="gru",
            bidirectional=True
        )
        self.dropout = nn.Dropout(p=args.dropout)
        self.LayerNorm = nn.LayerNorm(self.args.hidden_dim)

    def forward(self, input_tensors=None, attention_mask=None, position_ids=None, **kwargs):
        input_tensors = self.LayerNorm(input_tensors)
        output_tensors = self.op(input_tensors)
        output_tensors = self.dropout(output_tensors)
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


class LSTMWrapper(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layer, concat=False, bidir=True, dropout=0.3, return_last=True):
        super(LSTMWrapper, self).__init__()
        self.rnns = nn.ModuleList()
        for i in range(n_layer):
            if i == 0:
                input_dim_ = input_dim
                output_dim_ = hidden_dim
            else:
                input_dim_ = hidden_dim if not bidir else hidden_dim * 2
                output_dim_ = hidden_dim
            self.rnns.append(nn.LSTM(input_dim_, output_dim_, 1, bidirectional=bidir, batch_first=True))
        self.dropout = dropout
        self.concat = concat
        self.n_layer = n_layer
        self.return_last = return_last

    def forward(self, input, input_lengths=None):
        # input_length must be in decreasing order
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []

        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()

        for i in range(self.n_layer):
            output = F.dropout(output, p=self.dropout, training=self.training)

            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)

            output, _ = self.rnns[i](output)

            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen:  # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen - output.size(1), output.size(2))],
                                       dim=1)

            outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]


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
