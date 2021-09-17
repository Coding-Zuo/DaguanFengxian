# -*- coding:utf-8 -*-
import torch
import logging
import re

logger = logging.getLogger(__name__)


class FGM2(object):
    def __init__(self, model, emb_names=['word_embeddings', 'encoder.layer.0'], epsilon=1.0):
        self.model = model
        # emb_names这个参数 对应 embedding的参数名
        self.emb_names = emb_names
        self.epsilon = epsilon
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self):
        """Add adversity"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and re.search("|".join(self.emb_names), name):
                # 把真实值保存起来
                self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_adv = self.epsilon * param.grad / norm
                    param.data.add_(r_adv)

    def restore(self):
        """restore embedding"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and re.search("|".join(self.emb_names), name):
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if re.search("|".join(self.emb_names), name):
                    param.grad = self.grad_backup[name]
                else:
                    param.grad += self.grad_backup[name]


class PGD2(object):
    def __init__(self, model, emb_names=['word_embeddings', 'encoder.layer.0'], epsilon=1.0, alpha=0.3):
        self.model = model
        self.emb_names = emb_names
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        """Add adversity"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and re.search("|".join(self.emb_names), name):
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_adv = self.alpha * param.grad / norm
                    param.data.add_(r_adv)
                    param.data = self.project(name, param.data)

    def project(self, param_name, param_data):
        r_adv = param_data - self.emb_backup[param_name]
        if torch.norm(r_adv) > self.epsilon:
            r_adv_0 = self.epsilon * r_adv / torch.norm(r_adv)
        return self.emb_backup[param_name] + r_adv

    def restore(self):
        """restore embedding"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and re.search("|".join(self.emb_names), name):
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if re.search("|".join(self.emb_names), name):
                    param.grad = self.grad_backup[name]
                else:
                    param.grad += self.grad_backup[name]


class FGM1(object):
    """Reference: https://arxiv.org/pdf/1605.07725.pdf"""

    def __init__(self,
                 model,
                 emb_name='word_embeddings.',
                 epsilon=1.0):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self):
        """Add adversity."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        """restore embedding"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if self.emb_name in name:
                    param.grad = self.grad_backup[name]
                else:
                    param.grad += self.grad_backup[name]


class PGD1(object):
    """Reference: https://arxiv.org/pdf/1706.06083.pdf"""

    def __init__(self,
                 model,
                 emb_name='word_embeddings.',
                 epsilon=1.0,
                 alpha=0.3):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        """Add adversity."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)

    def restore(self):
        """restore embedding"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.epsilon:
            r = self.epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if self.emb_name in name:
                    param.grad = self.grad_backup[name]
                else:
                    param.grad += self.grad_backup[name]


adv_max_norm = 1.0
adv_norm_type = 'l2'
adv_init_mag = 0.03
adv_lr = 0.01


def getDelta(attention_mask, embeds_init):
    delta = None
    batch = embeds_init.shape[0]
    length = embeds_init.shape[-2]
    dim = embeds_init.shape[-1]

    attention_mask = attention_mask.view(-1, length)
    embeds_init = embeds_init.view(-1, length, dim)
    if adv_init_mag > 0:  # 影响attack首步是基于原始梯度(delta=0)，还是对抗梯度(delta!=0)
        input_mask = attention_mask.to(embeds_init)
        input_lengths = torch.sum(input_mask, 1)
        if adv_norm_type == "l2":
            delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
            dims = input_lengths * embeds_init.size(-1)
            mag = adv_init_mag / torch.sqrt(dims)
            delta = (delta * mag.view(-1, 1, 1)).detach()
        elif adv_norm_type == "linf":
            delta = torch.zeros_like(embeds_init).uniform_(-adv_init_mag, adv_init_mag)
            delta = delta * input_mask.unsqueeze(2)
    else:
        delta = torch.zeros_like(embeds_init)  # 扰动初始化
    # return delta.view(batch, -1, length, dim)
    return delta.view(batch, length, dim)


def updateDelta(delta, delta_grad, embeds_init):
    batch = delta.shape[0]
    length = delta.shape[-2]
    dim = delta.shape[-1]
    delta = delta.view(-1, length, dim)
    delta_grad = delta_grad.view(-1, length, dim)

    denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
    denorm = torch.clamp(denorm, min=1e-8)
    delta = (delta + adv_lr * delta_grad / denorm).detach()
    if adv_max_norm > 0:
        delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
        exceed_mask = (delta_norm > adv_max_norm).to(embeds_init)
        reweights = (adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
        #        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        delta = (delta * reweights).detach()
    # return delta.view(batch, -1, length, dim)
    return delta.view(batch, length, dim)


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.03, emb_name='bert.embeddings.word_embeddings.weight'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='bert.embeddings.word_embeddings.weight'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def fgm_use_bert_adv(fgm, model, batch, compute_loss, args):
    fgm.attack()
    start, end, q_type, paras, sents, ents, _, _ = model(batch, return_yp=True)
    loss_list = compute_loss(args, batch, start, end, paras, sents, ents, q_type)
    loss_list[0].backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
    fgm.restore()  # 恢复embedding参数


class PGD:
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=0.1, alpha=0.1, emb_name='bert.embeddings.word_embeddings.weight',
               is_first_attack=False):
        # emb_name 模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='bert.embeddings.word_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    param.grad = self.grad_backup[name]


def pgd_use_bert_adv(pgd, model, batch, compute_loss, args):
    pgd.backup_grad()
    # 对抗训练
    for t in range(args.k_pdg):
        pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
        if t != args.k_pdg - 1:
            model.zero_grad()
        else:
            pgd.restore_grad()
        start, end, q_type, paras, sents, ents, _, _ = model(batch, return_yp=True)[0]
        loss_list = compute_loss(args, batch, start, end, paras, sents, ents, q_type)
        loss_list[0].backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
    pgd.restore()  # 恢复embedding参数


class FreeLB(object):
    def __init__(self, adv_K=2, adv_lr=0.01, adv_init_mag=0.03, adv_max_norm=1.0, adv_norm_type='l2',
                 base_model='bert'):
        self.adv_K = adv_K
        self.adv_lr = adv_lr
        self.adv_max_norm = adv_max_norm
        self.adv_init_mag = adv_init_mag  # adv-training initialize with what magnitude, 即我们用多大的数值初始化delta
        self.adv_norm_type = adv_norm_type
        self.base_model = base_model

    def attack(self, model, inputs, gradient_accumulation_steps=1):
        input_ids = inputs['input_ids']
        if isinstance(model, torch.nn.DataParallel):
            embeds_init = getattr(model.module, self.base_model).embeddings.word_embeddings(input_ids)
        else:
            embeds_init = getattr(model, self.base_model).embeddings.word_embeddings(input_ids)
        if self.adv_init_mag > 0:  # 影响attack首步是基于原始梯度(delta=0)，还是对抗梯度(delta!=0)
            input_mask = inputs['attention_mask'].to(embeds_init)
            input_lengths = torch.sum(input_mask, 1)
            if self.adv_norm_type == "l2":
                delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                dims = input_lengths * embeds_init.size(-1)
                mag = self.adv_init_mag / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
            elif self.adv_norm_type == "linf":
                delta = torch.zeros_like(embeds_init).uniform_(-self.adv_init_mag, self.adv_init_mag)
                delta = delta * input_mask.unsqueeze(2)
        else:
            delta = torch.zeros_like(embeds_init)  # 扰动初始化
        loss, logits = None, None
        for astep in range(self.adv_K):
            delta.requires_grad_()
            inputs['inputs_embeds'] = delta + embeds_init  # 累积一次扰动delta
            ipids = inputs['input_ids']
            inputs['input_ids'] = None
            outputs = model(**inputs)
            loss, logits = outputs[:2]  # model outputs are always tuple in transformers (see doc)
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss = loss / gradient_accumulation_steps
            loss.backward()
            delta_grad = delta.grad.clone().detach()  # 备份扰动的grad
            if self.adv_norm_type == "l2":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()
                if self.adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > self.adv_max_norm).to(embeds_init)
                    reweights = (self.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                    delta = (delta * reweights).detach()
            elif self.adv_norm_type == "linf":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1,
                                                                                                         1)  # p='inf',无穷范数，获取绝对值最大者
                denorm = torch.clamp(denorm, min=1e-8)  # 类似np.clip，将数值夹逼到(min, max)之间
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()  # 计算该步的delta，然后累加到原delta值上(梯度上升)
                if self.adv_max_norm > 0:
                    delta = torch.clamp(delta, -self.adv_max_norm, self.adv_max_norm).detach()
            else:
                raise ValueError("Norm type {} not specified.".format(self.adv_norm_type))
            if isinstance(model, torch.nn.DataParallel):
                embeds_init = getattr(model.module, self.base_model).embeddings.word_embeddings(input_ids)
            elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
                embeds_init = getattr(model.module, self.base_model).embeddings.word_embeddings(input_ids)
            else:
                embeds_init = getattr(model, self.base_model).embeddings.word_embeddings(input_ids)
        return loss, logits


# https://github.com/lonePatient/TorchBlocks/blob/master/torchblocks/callback/adversarial.py
def freeLB_use_bert_adv(freelb, model, input_ids, attention_mask, token_type_ids, y, criterion, args):
    inputs = {
        "input_ids": input_ids,
        # "bbox": layout,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "masked_lm_labels": y
    }
    loss, prediction_scores = freelb.attack(model, inputs, args.accum_iter)
    return loss, prediction_scores

# if args.do_adv:
#     inputs = {
#         "input_ids": input_ids,
#         "bbox": layout,
#         "token_type_ids": segment_ids,
#         "attention_mask": input_mask,
#         "masked_lm_labels": lm_label_ids
#     }
#     loss, prediction_scores = freelb.attack(model, inputs)
# loss.backward()
# optimizer.step()
# scheduler.step()
# model.zero_grad()
