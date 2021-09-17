# -*- coding:utf-8 -*-
import numpy as np
import torch
from tqdm import tqdm
from torch.autograd import Variable


def replace_masked_values(tensor, mask, replace_with):
    """
    将``tensor``中的所有屏蔽值替换为``REPLACE_WITH``。``mask``必须是可广播的
    变成与``tensor``相同的形状。我们需要``tensor.dim()==mask.dim()``，否则我们
    不知道该解压mask的哪个维度。
    """
    if tensor.dim() != mask.dim():
        raise ValueError("tensor.dim() (%d) != mask.dim() (%d)" % (tensor.dim(), mask.dim()))
    return tensor.masked_fill((1 - mask).byte(), replace_with)


def weighted_sum(matrix, attention):
    """
    获取一个向量矩阵和矩阵中各行的一组权重(我们称之为“attention”向量)，并返回矩阵中行的加权和。这是典型的在注意力机制之后执行的计算。
    """
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)


def masked_softmax(vector, mask, dim=-1, memory_efficient=False, mask_fill_value=-1e32):
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # 为了限制来自掩码外的大型矢量元素的数值误差，我们将其置零。
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            # masked_vector = vector.masked_fill((1 - mask).bool(), mask_fill_value)
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.
    return torch.FloatTensor(sinusoid_table)


def get_embedding_matrix_and_vocab(w2v_file, skip_first_line=True, include_special_tokens=True):
    """
       Construct embedding matrix
    """
    embedding_dim = None

    # 先遍历一次，得到一个vocab list 和 向量list
    vocab_list = []
    vector_list = []

    with open(w2v_file, 'r', encoding='utf-8') as f_in:
        for i, line in tqdm(enumerate(f_in)):
            if skip_first_line:
                if i == 0: continue

            line = line.strip()
            if not line: continue

            line = line.split(" ")
            w_ = line[0]
            vec_ = line[1:]
            vec_ = [float(w.strip()) for w in vec_]

            if embedding_dim == None:
                embedding_dim = len(vec_)
            else:
                assert embedding_dim == len(vec_)

            vocab_list.append(w_)
            vector_list.append(vec_)

    # 添加两个特殊字符 PAD 和 UNK
    if include_special_tokens:
        vocab_list = ['pad', 'unk'] + vocab_list
        # 随机初始化两个向量
        pad_vec_ = (np.random.rand(embedding_dim).astype(np.float32) * 0.05).tolist()
        unk_vec_ = (np.random.rand(embedding_dim).astype(np.float32) * 0.05).tolist()
        vector_list = [pad_vec_, unk_vec_] + vector_list
    return vocab_list, vector_list
