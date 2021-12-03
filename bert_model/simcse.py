import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
import scipy.stats
import os
import tqdm
import random

# 生成有监督simcse需要的训练数据

def get_label_name2sents(train_data_path):
    dict_label_name2sents = {}

    with open(train_data_path, 'r', encoding='utf8') as fIn:
        for i, row in enumerate(fIn):
            row = row.strip()
            # 去除空行
            if not row:
                continue
            row = row.split(",")

            # 原始句子
            sent = row[1].strip()
            # 对应的标签
            label_name = row[2].strip()

            if label_name not in dict_label_name2sents:
                dict_label_name2sents[label_name] = set()
            # 将同标签的sent保存在一起
            dict_label_name2sents[label_name].add(sent)

    dict_label_name2sents_new = {}
    for label_name, sents in dict_label_name2sents.items():
        dict_label_name2sents_new[label_name] = list(sents)

    return dict_label_name2sents_new

label_list = [label.strip() for label in open('./data/user_data/process_data/splits/labels_file.txt', 'r',  \
                                                   encoding='utf-8')]
dict_label_name2sents_train = get_label_name2sents('./data/user_data/process_data/splits/fold_0/train.txt')

train_samples = []
for i in range(100000):
    label_0, label_1 = random.sample(label_list, 2)
    # 正样本和相似样本
    anchor, pos = random.sample(list(dict_label_name2sents_train[label_0]), 2)
    # hard negative样本
    neg = random.choice(list(dict_label_name2sents_train[label_1]))
    train_samples.append(
        {
            "sent0": anchor,
            "sent1": pos,
            "hard_neg": neg
        }
    )


np.random.shuffle(train_samples)
df_train_samples = pd.DataFrame(
    train_samples[: int(len(train_samples) * 0.9)]
)
os.makedirs('./data/user_data/process_data/nli_data', exist_ok=True)

df_train_samples.to_csv(
    os.path.join('./data/user_data/process_data/nli_data', "train.txt"),
    index=False,
    sep="\t"
)
df_dev_samples = pd.DataFrame(
    train_samples[int(len(train_samples) * 0.9): ]
)
df_dev_samples.to_csv(
    os.path.join('./data/user_data/process_data/nli_data', "test.txt"),
    index=False,
    sep="\t"
)

# 通过有监督的simcse来微调预训练的bert模型
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_path = './data/user_data/bert_model'
save_path = './data/user_data/simcse_model'
tokenizer = BertTokenizer.from_pretrained(model_path)
Config = BertConfig.from_pretrained(model_path, gradient_checkpointing=True)

for f in open(os.path.join(model_path, 'vocab.txt'), 'r', encoding='utf-8'):
    with open(os.path.join(save_path, 'vocab.txt'), 'w+', encoding='utf-8') as f_out:
        f_out.write(f.readline().strip() + '\n')



output_way = 'pooler'
batch_size = 64
learning_rate = 0.5e-6
maxlen = 132

nli_file_path = './data/user_data/process_data/nli_data'
nli_train_file = 'train.txt'
nli_test_file = 'test.txt'


def load_snli_vocab(path):
    data = []
    a = 0
    with open(path, 'r', encoding='utf-8') as f:
        for i in f:
            a += 1
            if a == 1:
                continue
            data.append(i)
    return data

snil_vocab = load_snli_vocab(os.path.join(nli_file_path, nli_train_file))
np.random.shuffle(snil_vocab)

test_data = load_snli_vocab(os.path.join(nli_file_path, nli_test_file))
np.random.shuffle(test_data)


class TrainDataset(Dataset):
    def __init__(self, data, tokenizer, maxlen, transform=None, target_transform=None):
        self.data = data
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.transform = transform
        self.target_transform = target_transform

    def text_to_id(self, source):
        text = source.split('\t')
        origin = text[0].strip()
        entailment = text[1].strip()
        contradiction = text[2].strip()
        sample = self.tokenizer([origin, entailment, contradiction], max_length=self.maxlen, truncation=True,
                                padding='max_length', return_tensors='pt')
        return sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.text_to_id(self.data[idx])


class TestDataset:
    def __init__(self, data, tokenizer, maxlen):
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.traget_idxs = self.text_to_id([x[0] for x in data])
        self.source_idxs = self.text_to_id([x[1] for x in data])
        self.label_list = [int(x[2]) for x in data]
        assert len(self.traget_idxs['input_ids']) == len(self.source_idxs['input_ids'])

    def text_to_id(self, source):
        sample = self.tokenizer(source, max_length=self.maxlen, truncation=True, padding='max_length',
                                return_tensors='pt')
        return sample

    def get_data(self):
        return self.traget_idxs, self.source_idxs, self.label_list


class NeuralNetwork(nn.Module):
    def __init__(self, model_path, output_way):
        super(NeuralNetwork, self).__init__()
        self.bert = BertModel.from_pretrained(model_path, config=Config)
        self.output_way = output_way
        assert output_way in ['cls', 'pooler']

    def forward(self, input_ids, attention_mask, token_type_ids):
        x1 = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if self.output_way == 'cls':
            output = x1.last_hidden_state[:, 0]
        elif self.output_way == 'pooler':
            output = x1.pooler_output
        return output


model = NeuralNetwork(model_path, output_way).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=270000,
            power=2
        )

training_data = TrainDataset(snil_vocab, tokenizer, maxlen)
train_dataloader = DataLoader(training_data, batch_size=batch_size)

testing_data = TrainDataset(test_data, tokenizer, maxlen)
test_dataloader = DataLoader(testing_data, batch_size=64)

def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation

def compute_loss(y_pred, lamda=0.05):
    # row为正样本的索引
    row = torch.arange(0, y_pred.shape[0], 3, device='cuda')
    col = torch.arange(y_pred.shape[0], device='cuda')
    # torch.where()返回一个tuple，col为相似样本和hard neg样本的索引
    col = torch.where(col % 3 != 0)[0].cuda()
    y_true = torch.arange(0, len(col), 2, device='cuda')
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    # torch自带的快速计算相似度矩阵的方法
    similarities = torch.index_select(similarities, 0, row)
    similarities = torch.index_select(similarities, 1, col)
    # 屏蔽对角矩阵即自身相等的loss
    similarities = similarities / lamda
    # 论文中除以 temperature 超参 0.05
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)

def train(dataloader, test_dataloader, model, optimizer, epochs):
    the_best_test_loss = 1000
    losses = []
    early_stop = 6
    size = len(dataloader.dataset)
    stop_epoch = 0
    global_batch = 0

    for epoch in range(epochs):

        for batch, data in enumerate(tqdm.tqdm(dataloader)):
            model.train()
            input_ids = data['input_ids'].view(len(data['input_ids']) * 3, -1).to(device)  # batch_size*3, max_len
            attention_mask = data['attention_mask'].view(len(data['attention_mask']) * 3, -1).to(device)
            token_type_ids = data['token_type_ids'].view(len(data['token_type_ids']) * 3, -1).to(device)
            pred = model(input_ids, attention_mask, token_type_ids)
            loss = compute_loss(pred)
            # 反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            scheduler.step()
            global_batch += 1
            if batch % 100 == 0 and batch != 0:
                loss_mean, current = sum(losses) / batch, batch * int(len(input_ids) / 3)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            if global_batch % 300 == 0:
                with torch.no_grad():
                    losses_test = []
                    for batch_test, data_test in enumerate(tqdm.tqdm(test_dataloader)):
                        input_ids = data_test['input_ids'].view(len(data_test['input_ids']) * 3, -1).to(device)  # batch_size*3, max_len
                        attention_mask = data_test['attention_mask'].view(len(data_test['attention_mask']) * 3, -1).to(device)
                        token_type_ids = data_test['token_type_ids'].view(len(data_test['token_type_ids']) * 3, -1).to(device)
                        pred_test = model(input_ids, attention_mask, token_type_ids)
                        loss_test = compute_loss(pred_test)
                        losses_test.append(loss_test.item())
                    loss_test_mean = sum(losses_test) / (batch_test + 1)
                    if loss_test_mean < the_best_test_loss:
                        the_best_test_loss = loss_test_mean
                        print('Saving the model')
                        model.bert.save_pretrained(save_path)
                        print("The best loss of test{}".format(the_best_test_loss))
                        stop_epoch = 0
                    else: stop_epoch += 1
                    if stop_epoch > early_stop:
                        break

if __name__ == '__main__':
    epochs = 50
    train(train_dataloader, test_dataloader, model, optimizer, epochs)
